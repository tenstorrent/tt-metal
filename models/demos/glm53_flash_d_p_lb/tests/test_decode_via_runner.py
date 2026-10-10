# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode via prefill through tt-metal's prefill runner (the process the server feeds): GLM-5.3-Flash on the LoudBox
(2x4, all 45 layers) in models/demos/glm53_flash_d_p_lb/serve/runner.py, which is prefill_runner.main plus a logits
file per chunk. This test is the feeder, in the place of the server's engine: it pushes each chunk over the runner's
H2D stream service (uint32 [sp, 1, chunk / sp] + the 12-byte {slot, start, end} header, as tt-metal's prefill
producer does), drains the chunk's layer acks (one per DSA layer) and reads the logits file; every generated token
is the prefill of the chunk holding it, from that chunk's boundary.

  smoke:        the spec's smoke prompt, greedy, slot 0; the answer must contain the expected word.
  interleave:   two prompts decoded alternately in slots 0 and 1, one of them across a chunk boundary (the KDA
                snapshot restore); each must give the same tokens as its run alone (slot 0 tokens from the smoke).
  consistency:  at the steps that restore a boundary snapshot, the decode logits vs a cold prefill of the same
                tokens in the other slot (argmax equal, PCC >= 0.999).

  BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \\
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_decode_via_runner.py -s

The test opens no device: the runner subprocess does. Env: GLM_DVP_CHUNK (2048), GLM_DVP_MAX_SEQ (8192),
GLM_DVP_SMOKE_TOKENS (16), GLM_DVP_STEPS (6), GLM_DVP_READY_S (5400: the runner's load + compile)."""

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.common.bringup.testing.harness import spec

S = spec()
CHUNK = int(os.environ.get("GLM_DVP_CHUNK", "2048"))
MAX_SEQ = int(os.environ.get("GLM_DVP_MAX_SEQ", "8192"))
SMOKE_TOKENS = int(os.environ.get("GLM_DVP_SMOKE_TOKENS", "16"))
STEPS = int(os.environ.get("GLM_DVP_STEPS", "6"))
READY_S = float(os.environ.get("GLM_DVP_READY_S", "5400"))
STEP_S = 600.0
PAD_ID = 0xFFFFFFFF
SP, TP = S.mesh
ROOT = Path(__file__).resolve().parents[4]


def runner_env(out: Path, sid: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    env.update(
        PREFILL_MODEL=S.get("contract.adapter", S.model),
        PREFILL_SP=str(SP),
        PREFILL_TP=str(TP),
        PREFILL_NUM_LAYERS=str(S.num_layers),
        PREFILL_CHUNK_SIZE=str(CHUNK),
        PREFILL_MAX_SEQ_LEN=str(MAX_SEQ),
        PREFILL_NUM_USERS="2",
        PREFILL_FABRIC_MODE="2d",
        PREFILL_LAYER_ACK_D2H="0",  # the runtime acks through the host sink (one per DSA layer, after an event sync)
        PREFILL_USE_TRACE="0",
        PREFILL_ENABLE_MIGRATION="0",
        PREFILL_H2D_SERVICE_ID=sid,
        PREFILL_H2D_CONNECT_TIMEOUT="600",
        GLM_SERVE_LOGITS_DIR=str(out / "logits"),
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    env.setdefault("OMP_NUM_THREADS", "16")
    # the runner waits on its H2D stream between chunks (the feeder samples, reads logits): not a hang
    env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "900"
    return env


class Feeder:
    """The server's side of the runner: H2D pushes, layer acks, logits files."""

    def __init__(self, out: Path, sid: str, proc):
        import ttnn

        self.out, self.proc, self.ttnn = out, proc, ttnn
        self.logits = out / "logits"
        t0 = time.time()
        while True:  # the service exists once the runner has loaded, compiled and entered its request loop
            self._alive()
            try:
                self.service = ttnn.H2DStreamService.connect(sid, timeout_ms=10000)
                break
            except Exception:
                if time.time() - t0 > READY_S:
                    raise RuntimeError(f"runner not serving after {READY_S:.0f} s (see {out / 'runner.log'})")
        self.payload = self.service.payload_size_bytes()
        self.acks = ttnn.InterProcessCounterChannel.connect(f"/tt_prefill_layer_acks_{sid}", connect_timeout_ms=60000)
        from models.demos.common.prefill.adapter import get_adapter

        self.acks_per_chunk = get_adapter(S.get("contract.adapter", S.model)).num_kv_cache_layers(S.num_layers)
        print(f"[dvp] runner serving after {time.time() - t0:.0f} s; {self.acks_per_chunk} acks per chunk", flush=True)

    def _alive(self):
        if self.proc.poll() is not None:
            raise RuntimeError(f"runner exited {self.proc.returncode} (see {self.out / 'runner.log'})")

    def _push(self, ids, slot, start, end):
        rows = list(ids) + [PAD_ID] * (CHUNK - len(ids))
        payload = np.asarray(rows, dtype=np.uint32).reshape(SP, 1, CHUNK // SP)
        assert payload.nbytes == self.payload, (payload.nbytes, self.payload)
        self.service.forward_to_tensor_bytes(payload, metadata=struct.pack("<3I", slot, start, end))

    def chunk(self, seq, slot, start, end):
        """Prefill seq[start:end] in the slot -> logits after seq[end - 1]."""
        self._push(seq[start:end], slot, start, end)
        got, t0 = 0, time.time()
        while got < self.acks_per_chunk:
            got += self.acks.try_consume_all()
            self._check(t0)
        while True:
            files = sorted(self.logits.glob(f"s{slot}_e{end}_*.bin"), key=lambda p: int(p.stem.split("_")[-1]))
            if files:
                break
            self._check(t0)
        raw = files[-1].read_bytes()
        nl = raw.index(b"\n")
        head = json.loads(raw[:nl])
        for f in files:
            f.unlink()
        assert head["start"] == start and head["finite"], head
        return torch.from_numpy(np.frombuffer(raw[nl + 1 :], dtype=np.float32).copy())

    def _check(self, t0):
        self._alive()
        if time.time() - t0 > STEP_S:
            raise RuntimeError(f"no answer for a chunk after {STEP_S:.0f} s (see {self.out / 'runner.log'})")
        time.sleep(0.005)

    def prefill(self, ids, slot):
        logits = None
        for start in range(0, len(ids), CHUNK):
            logits = self.chunk(ids, slot, start, min(start + CHUNK, len(ids)))
        return logits

    def shutdown(self):
        self._push([], 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)  # the runner's sentinel (metadata -1, -1, -1)


class Gen:
    """One greedy generation, stepped one token at a time (so two can interleave)."""

    def __init__(self, feeder, ids, slot, eos=None):
        self.f, self.ids, self.slot, self.eos = feeder, list(ids), slot, eos
        self.out, self.steps, self.times = [], [feeder.prefill(self.ids, slot)], []

    def step(self):
        t = int(self.steps[-1].argmax())
        self.out.append(t)
        if t == self.eos:
            return False
        seq = self.ids + self.out
        t0 = time.time()
        self.steps.append(self.f.chunk(seq, self.slot, (len(seq) - 1) // CHUNK * CHUNK, len(seq)))
        self.times.append(time.time() - t0)
        return True


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_decode_via_runner(tmp_path):
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        pytest.skip("starts the runner process: not in the precompile pass")
    from models.demos.common.bringup.testing.smoke import prompt_ids, tokenizer

    tok = tokenizer(S)
    out = tmp_path
    sid = f"glm_dvp_{os.getpid()}"
    log = (out / "runner.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "models.demos.glm53_flash_d_p_lb.serve.runner"],
        env=runner_env(out, sid),
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True,
    )
    fails = []
    try:
        f = Feeder(out, sid, proc)
        # smoke, slot 0
        ids = prompt_ids(S, tok)
        expect = S.data["intake"]["smoke"]["expect"]
        g = Gen(f, ids, 0, tok.eos_token_id)
        while len(g.out) < SMOKE_TOKENS and g.step() and expect.lower() not in tok.decode(g.out).lower():
            pass
        text = tok.decode(g.out)
        st = sum(g.times) / max(1, len(g.times))
        print(f"[dvp] smoke: -> {text!r} ({len(g.out)} tokens, {st:.2f} s/token through the runner)", flush=True)
        if expect.lower() not in text.lower():
            fails.append(f"smoke: expected {expect!r} in {text!r}")
        smoke_tokens = list(g.out)

        # long prompt alone (slot 1), across the chunk boundary
        filler = (
            "The lighthouse keeper kept a log of every ship that passed, the weather, the tide and the colour of "
            "the sea. "
        )
        long_ids = tok("Continue this logbook in the same style.\n\n" + filler * 400, add_special_tokens=False)[
            "input_ids"
        ][: 2 * CHUNK - 3]
        alone = Gen(f, long_ids, 1)
        for _ in range(STEPS):
            alone.step()
        print(f"[dvp] long alone: {tok.decode(alone.out)!r}, {sum(alone.times) / len(alone.times):.2f} s/token")
        # consistency: steps that restore the boundary snapshot (1: at C, 5: at 2 C) and the one ending a chunk (3)
        for k in (1, 3, 5):
            ref = f.prefill(long_ids + alone.out[:k], 0)
            p, same = _pcc(alone.steps[k], ref), int(alone.steps[k].argmax()) == int(ref.argmax())
            print(f"[dvp] step {k} (length {len(long_ids) + k}): decode vs cold prefill PCC {p:.6f}, argmax {same}")
            if p < 0.999 or not same:
                fails.append(f"consistency step {k}: PCC {p:.6f}, argmax equal {same}")

        # interleave: the smoke in slot 0 and the long prompt in slot 1, one step each in turn
        a, b = Gen(f, ids, 0, tok.eos_token_id), Gen(f, long_ids, 1)
        for _ in range(max(len(smoke_tokens), STEPS)):
            if len(a.out) < len(smoke_tokens):
                a.step()
            if len(b.out) < STEPS:
                b.step()
        print(f"[dvp] interleaved: slot 0 {tok.decode(a.out)!r}, slot 1 {tok.decode(b.out)!r}")
        if a.out[: len(smoke_tokens)] != smoke_tokens:
            fails.append(f"interleave slot 0: {a.out} != alone {smoke_tokens}")
        if b.out[:STEPS] != alone.out[:STEPS]:
            fails.append(f"interleave slot 1: {b.out} != alone {alone.out}")
        f.shutdown()
        proc.wait(timeout=300)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        print(f"[dvp] runner log: {out / 'runner.log'}")
    assert not fails, fails
