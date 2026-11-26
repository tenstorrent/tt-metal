# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the verify penalties of Qwen36MTPContractForCausalLM (tt/spec_penalties.py VerifyPenalties) on controlled
logits, as the model runs them: persistent buffers of each device's vocabulary shard [1, 1, rows, V / TP] bfloat16
(V = 248320), the bodies warmed eagerly and captured as traces with program cache misses disallowed, the counts written
by the eager rebuild and add passes.

* The penalty trace over the B-user rows and over a verify row bucket's rows: column j of every penalty row is vLLM's
  apply_penalties over the committed output plus drafts 1 .. j (spec_penalties.reference_penalties), rounded once to
  bfloat16; every token of a neutral row, and every token a frequency or presence penalty does not touch, is bit for bit
  what was written; a greedy penalty row's verify ids are the float32 argmax of its penalized logits (the plugin's host
  path), across the shards; the other lanes keep their ids.
* The commit trace of a chained verify adds each tracked user's commit (its ids, or its draws, up to the drafter pack's
  count) and leaves the other users.

    pytest models/demos/blackhole/qwen36/tests/test_spec_penalties.py    (Blackhole; MESH_DEVICE=P150x8)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    StepPenalties,
    VerifyPenalties,
    counts_of,
    reference_penalties,
)
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, draft_anchor, drafter_pack_rows, pack_width

VOCAB = 248320
MESH = _resolve_mesh_shape(8)
B, T = 8, 4
R = B * T
BUCKET = 2  # users of the verify row bucket whose trace runs too
# user -> (repetition, frequency, presence, greedy): 0-2 one penalty each, 3 all three, 4 a repetition below 1;
# 3 and 4 are drawn rows (their ids stay), 5-7 neutral.
PENALTIES = {
    0: (1.3, 0.0, 0.0, True),
    1: (1.0, 0.4, 0.0, True),
    2: (1.0, 0.0, 0.5, True),
    3: (1.15, 0.2, 0.3, False),
    4: (0.8, 0.0, 0.0, False),
}


def _parametrize(fn):
    # The fused argmax all-gathers each shard's maximum.
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _replicated(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _write(mesh, host, dev, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mapper=None):
    mapper = ttnn.ReplicateTensorToMesh(mesh) if mapper is None else mapper
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=mapper), dev)


def _logits_buffer(mesh, rows):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=_mapper(mesh),
    )


def _read_logits(mesh, buf):
    rows = int(buf.shape[2])
    if _shards(mesh) > 1:
        return ttnn.to_torch(buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).reshape(rows, VOCAB)
    return ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).reshape(rows, VOCAB)


def _lanes(t, n=32):
    """[devices, n] words of the [1, 1, 1, >= n] uint32 ``t``."""
    return torch.stack([ttnn.to_torch(d).reshape(-1)[:n].to(torch.int64) for d in ttnn.get_device_tensors(t)])


def _capture(mesh, fn):
    mesh.set_program_cache_misses_allowed(False)
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    ttnn.synchronize_device(mesh)
    return trace


def _bf16_ulps(a, b):
    """|a - b| in units of b's bfloat16 ulp (a, b float32 holding bfloat16 values)."""
    scale = torch.pow(2.0, torch.floor(torch.log2(b.abs().clamp_min(1e-30))) - 7)
    return ((a - b).abs() / scale).max().item() if a.numel() else 0.0


class _Rig:
    """A VerifyPenalties over persistent stand-ins of what the model binds it to: the verify step buffer, the verify
    ids lanes, the sampler's draws, the drafter pack and the vocabulary shard of the B-user and the bucket verify."""

    def __init__(self, mesh):
        self.mesh = mesh
        fused = FusedShardArgmax(mesh, VOCAB, _shards(mesh))
        self.pen = VerifyPenalties(mesh, B, T, VOCAB, shards=_shards(mesh), argmax=lambda x: fused(x, LANES))
        self.step = _replicated(mesh, torch.zeros(VERIFY_STEP_ROWS, pack_width(R), dtype=torch.int32), ttnn.uint32)
        self.ids = _replicated(mesh, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        self.draws = _replicated(mesh, torch.zeros(1, 1, 1, R, dtype=torch.int32), ttnn.uint32)
        self.pack = _replicated(
            mesh, torch.zeros(drafter_pack_rows(T - 1), pack_width(R), dtype=torch.int32), ttnn.uint32
        )
        self.logits = {R: _logits_buffer(mesh, R), BUCKET * T: _logits_buffer(mesh, BUCKET * T)}
        self.pen.bind(step=self.step, ids_out=self.ids, draws=self.draws, pack=self.pack)
        for buf in self.logits.values():
            self.pen.apply_body(buf)
        self.pen.commit_body()
        ttnn.synchronize_device(mesh)
        self.pen.warm_eager()
        self.apply = {rows: _capture(mesh, lambda b=buf: self.pen.apply_body(b)) for rows, buf in self.logits.items()}
        self.commit = _capture(mesh, self.pen.commit_body)

    def stage_tokens(self, tokens):
        step = torch.zeros(VERIFY_STEP_ROWS, pack_width(R), dtype=torch.int32)
        step[0, :R] = tokens.reshape(-1).to(torch.int32)
        _write(self.mesh, step, self.step, ttnn.uint32)

    def replay(self, trace):
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
        finally:
            self.mesh.set_program_cache_misses_allowed(True)

    def release(self):
        for trace in (*self.apply.values(), self.commit):
            ttnn.release_trace(self.mesh, trace)
        self.pen.release()


def _step_penalties():
    rep, freq, pres = torch.ones(B), torch.zeros(B), torch.zeros(B)
    for u, (r, f, p, _) in PENALTIES.items():
        rep[u], freq[u], pres[u] = r, f, p
    live = torch.tensor([u in PENALTIES for u in range(B)])
    return StepPenalties(rep, freq, pres, live)


def _greedy_lanes():
    return [int(PENALTIES.get(r // T, (0, 0, 0, False))[3]) for r in range(R)]


@_parametrize
def test_the_penalty_trace_scores_every_column_as_vllm_does(mesh_device):
    rig = _Rig(mesh_device)
    g = torch.Generator().manual_seed(4)
    try:
        prompts = {
            u: torch.randint(0, VOCAB, (n,), generator=g) for u, n in zip(range(B), (300, 9, 40, 1000, 77, 5, 5, 5))
        }
        outputs = {
            u: torch.randint(0, VOCAB, (n,), generator=g) for u, n in zip(range(B), (60, 400, 25, 33, 12, 3, 3, 3))
        }
        for u in range(B):  # a frequent token, and one repeated in the prompt
            outputs[u][:10] = outputs[u][0]
        tokens = torch.randint(0, VOCAB, (B, T), generator=g)
        tokens[:, 0] = torch.stack([outputs[u][-1] for u in range(B)])
        tokens[0, 2] = outputs[0][0]  # a draft already in the history
        tokens[1, 3] = tokens[1, 0]  # a padded column repeats column 0
        rig.pen.rebuild({u: (prompts[u], outputs[u]) for u in range(B)})
        rig.pen.stage(_step_penalties(), _greedy_lanes())
        rig.stage_tokens(tokens)
        for rows in (R, BUCKET * T):
            raw = (torch.randn(rows, VOCAB, generator=g) * 3).to(torch.bfloat16)
            for r in range(rows):  # each row's maximum is a token its penalties lower
                u = r // T
                raw[r, int(outputs[u][0])] = raw[r].max() + 0.25
                raw[r, int(prompts[u][0])] = raw[r].max() + 0.125
            _write(mesh_device, raw.reshape(1, 1, rows, VOCAB), rig.logits[rows], ttnn.bfloat16, ttnn.TILE_LAYOUT,
                   _mapper(mesh_device))  # fmt: skip
            ids_before = torch.arange(1000, 1032, dtype=torch.int32).reshape(1, 1, 1, 32)
            _write(mesh_device, ids_before, rig.ids, ttnn.uint32)
            rig.replay(rig.apply[rows])
            got = _read_logits(mesh_device, rig.logits[rows]).to(torch.float32)
            lanes = _lanes(rig.ids)
            assert bool((lanes == lanes[0]).all()), "the devices of the mesh hold different ids"
            worst, exact, greedy_flips = 0.0, 0, 0
            for r in range(rows):
                u, j = divmod(r, T)
                x = raw[r].to(torch.float32)
                if u not in PENALTIES:
                    assert torch.equal(got[r], x), f"neutral row {r} changed"
                    assert int(lanes[0, r]) == 1000 + r
                    continue
                rep, freq, pres, greedy = PENALTIES[u]
                history = torch.cat([outputs[u], tokens[u, 1 : j + 1]])
                want32 = reference_penalties(x, prompts[u], history, rep, freq, pres)
                want = want32.to(torch.bfloat16).to(torch.float32)
                touched = counts_of(history, VOCAB) > 0
                if rep != 1.0:
                    touched |= counts_of(prompts[u], VOCAB) > 0
                assert torch.equal(got[r, ~touched], x[~touched]), f"row {r}: a token no penalty touches changed"
                worst = max(worst, _bf16_ulps(got[r, touched], want[touched]))
                exact += int(torch.equal(got[r], want))
                if greedy:
                    assert int(lanes[0, r]) == int(want32.argmax()), f"row {r}: not the float32 argmax"
                    greedy_flips += int(want.argmax()) != int(want32.argmax())
                else:
                    assert int(lanes[0, r]) == 1000 + r, f"drawn row {r}: its ids lane changed"
            penalized = sum(1 for r in range(rows) if r // T in PENALTIES)
            logger.info(
                f"[penalties] {rows} rows: {exact} of {penalized} penalized rows bit-exact to the bfloat16 reference, "
                f"worst {worst:.3f} bf16 ulp; {greedy_flips} greedy rows whose bfloat16 argmax would differ"
            )
            assert worst <= 1.0, f"a penalized logit is {worst} bf16 ulp off the reference"
    finally:
        rig.release()


@_parametrize
def test_the_commit_trace_adds_each_tracked_users_commit(mesh_device):
    rig = _Rig(mesh_device)
    g = torch.Generator().manual_seed(5)
    try:
        outputs = {u: torch.randint(0, VOCAB, (20,), generator=g) for u in range(B)}
        rig.pen.rebuild({u: (torch.tensor([], dtype=torch.int64), outputs[u]) for u in range(B)})
        ids = torch.randint(0, VOCAB, (B, T), generator=g)
        draws = torch.randint(0, VOCAB, (B, T), generator=g)
        m = [u % T for u in range(B)]
        lanes = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        lanes[0, 0, 0, :R] = ids.reshape(-1).to(torch.int32)
        _write(mesh_device, lanes, rig.ids, ttnn.uint32)
        _write(mesh_device, draws.reshape(1, 1, 1, R).to(torch.int32), rig.draws, ttnn.uint32)
        pack = torch.zeros(drafter_pack_rows(T - 1), pack_width(R), dtype=torch.int32)
        pack[draft_anchor(T - 1), :B] = torch.tensor([u * T + m[u] for u in range(B)], dtype=torch.int32)
        _write(mesh_device, pack, rig.pack, ttnn.uint32)
        tracked = [u < 6 for u in range(B)]
        drawn = [int(u % 2) for u in range(B)]
        rig.pen.stage(_step_penalties(), [0] * R, track=tracked, drawn=drawn)
        rig.replay(rig.commit)
        counts, _ = rig.pen.read(range(B))
        for u in range(B):
            commit = (draws if drawn[u] else ids)[u, : m[u] + 1]
            history = torch.cat([outputs[u], commit]) if tracked[u] else outputs[u]
            assert torch.equal(counts[u], counts_of(history, VOCAB)), f"user {u}"
        rig.pen.add({0: torch.tensor([7, 7, 248319])})
        assert int(rig.pen.read([0])[0][0, 7]) == int(counts[0, 7]) + 2
    finally:
        rig.release()


# ------------------------------------------------------------------------------------------------- the model, served
K = 3
MAX_MODEL_LEN = 4096
KINDS = {
    "repetition": dict(repetition_penalty=1.3),
    "frequency": dict(frequency_penalty=0.6),
    "presence": dict(presence_penalty=0.8),
}


class _Runner:
    """B = 1 calls into Qwen36MTPContractForCausalLM as the plugin runner makes them for a greedy request with
    penalties. ``device``: every step is an argmax_ids verify of a device-sampled step (the device applies the
    penalties); else every step is a logits verify and the host applies vLLM's penalties to each column after the
    committed output and the drafts before it (the plugin's logits walk), then takes the argmax. ``drafts``: verify the
    drafter's proposals; without them every verify is one column, so both paths run the same computation."""

    def __init__(self, cls, kv, layout, seed):
        from models.demos.blackhole.qwen36.tests.spec_block_helpers import SpecBlocks

        self.cls, self.kv = cls, kv
        self.bs = layout.block_size
        self.width = -(-MAX_MODEL_LEN // self.bs)
        g = torch.Generator().manual_seed(seed)
        free = (torch.randperm(layout.num_blocks - 1, generator=g) + 1).tolist()
        self.state = SpecBlocks(layout, K, "none", free=free[: layout.num_groups * (K + 1)])
        self.free, self.blocks = free[layout.num_groups * (K + 1) :], []

    def _table(self, last):
        need = min(self.width, -(-(last + 2 * (K + 1)) // self.bs))
        new = []
        while len(self.blocks) < need:
            new.append(self.free.pop())
            self.blocks.append(new[-1])
        if new:
            self.cls.zero_kv_cache_blocks(self.kv, new)
        return torch.tensor([self.blocks + [0] * (self.width - len(self.blocks))], dtype=torch.int32)

    def run(self, prompt, max_new, penalty, device, drafts):
        import time
        from types import SimpleNamespace

        cls, P = self.cls, len(prompt)
        rep = penalty.get("repetition_penalty", 1.0)
        freq = penalty.get("frequency_penalty", 0.0)
        pres = penalty.get("presence_penalty", 0.0)
        bmax = cls.model[0].args.max_batch_size

        def pick(x, history):
            return int(reference_penalties(x.float(), prompt, history, rep, freq, pres).argmax())

        self.state.admit(0)
        logits, _ = cls.prefill_forward(
            torch.tensor([prompt], dtype=torch.int32),
            self._table(P - 1),
            self.kv,
            prompt_lens=torch.tensor([P]).numpy(),
            start_pos=torch.tensor([0]).numpy(),
            empty_slots=[0],
            enable_trace=True,
            mamba_state=self.state.state({0: self.state.prefill(0, 0, P)}, bmax, intermediate=()),
        )
        out, proposal, acc, seconds, steps = [pick(logits[0, -1], [])], [], 1, 0.0, 0
        sampling = SimpleNamespace(
            temperature=[0.0], top_k=[1], top_p=[1.0], seed=[None], repetition_penalty=[rep],
            frequency_penalty=[freq], presence_penalty=[pres],
        )  # fmt: skip
        while len(out) < max_new:
            pos0 = P + len(out) - 1
            n = len(proposal) if drafts else 0
            tok = torch.full((1, 1 + K), -1, dtype=torch.int32)
            pos = torch.full((1, 1 + K), -1, dtype=torch.int32)
            tok[0, : 1 + n] = torch.tensor([out[-1]] + proposal[:n], dtype=torch.int32)
            pos[0, : 1 + n] = torch.arange(pos0, pos0 + 1 + n, dtype=torch.int32)
            kw = dict(
                tokens=tok,
                start_pos=pos,
                page_table=self._table(pos0),
                kv_cache=self.kv,
                enable_trace=True,
                read_from_device=False,
                num_valid_drafts=torch.tensor([n], dtype=torch.int32),
                accepted_counts=torch.tensor([acc], dtype=torch.int32),
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
                mamba_state=self.state.state({0: self.state.decode(0, pos0, 1 + n)}, bmax),
            )
            t0 = time.perf_counter()
            if device:
                res = cls.decode_forward(
                    spec_mode="argmax_ids",
                    sampling_params=sampling,
                    prompt_tokens=torch.tensor([prompt], dtype=torch.int32),
                    output_tokens=torch.tensor([out], dtype=torch.int32),
                    **kw,
                )
                ids = cls.read_decode_output(res.argmax_ids)[0].tolist()
            else:
                res = cls.decode_forward(spec_mode="logits", **kw)
                ids = [pick(res.logits[0, j], out + proposal[:j]) for j in range(n + 1)]
            seconds += time.perf_counter() - t0
            steps += 1
            j = 0
            while j < n and proposal[j] == ids[j]:
                j += 1
            new = (proposal[:j] + [ids[j]])[: max_new - len(out)]
            out.extend(new)
            acc = len(new)
            self.state.commit(0, acc)
            committed = torch.full((1, 1 + K), -1, dtype=torch.int32)
            cpos = torch.full((1, 1 + K), -1, dtype=torch.int32)
            committed[0, :acc] = torch.tensor(new, dtype=torch.int32)
            cpos[0, :acc] = torch.arange(pos0 + 1, pos0 + 1 + acc, dtype=torch.int32)
            d = cls.propose_draft_tokens(K, committed, cpos, torch.tensor([acc], dtype=torch.int32), hidden=res.hidden)
            proposal = [int(t) for t in d.draft_token_ids[0, : int(d.num_valid[0])]]
        cls.release_request(0)
        self.state.release(0)
        return out, steps, seconds


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_greedy_penalised_request_commits_the_host_paths_tokens(mesh_device):
    """Qwen3.8 on the class the plugin serves, warmed for device sampling: a greedy request with each penalty kind
    decodes the same tokens on the device path (argmax_ids verifies with the penalties applied on the device) as on
    the host path (logits verifies, vLLM's penalties and the argmax on the host), one column per verify; with drafts
    the device path still matches, up to a near tie the different verify grouping can flip. The device counts end equal
    to the request's committed output. Logs the time per verify of both paths."""
    import math
    import os
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.demo.text_demo import _get_prompt
    from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

    if not (os.environ.get("HF_MODEL") or os.environ.get("MODEL_WEIGHTS_DIR")):
        pytest.skip("set HF_MODEL (checkpoint dir)")
    if _shards(mesh_device) == 1:
        pytest.skip("the verify penalties are on the TP path")
    mesh_device.enable_program_cache()
    hf_config = SimpleNamespace(_name_or_path=os.environ.get("HF_MODEL"))
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=MAX_MODEL_LEN, tt_data_parallel=1, optimizations=None
    )
    model = cls.model[0]
    probe = PluginLayout.for_model(model, num_blocks=1)
    width = math.ceil(MAX_MODEL_LEN / probe.block_size)
    layout = PluginLayout.for_model(model, num_blocks=width + probe.num_groups * (K + 1) + 8)
    kv = cls.allocate_kv_cache_per_layer(layout.per_layer_specs)
    for enable_trace in (False, True):
        cls.warmup_model_prefill(enable_trace=enable_trace, kv_cache=kv, can_sample_on_device=True)
        cls.warmup_model_decode(
            enable_trace=enable_trace, kv_cache=kv, max_batch_size=1, num_blocks=width, can_sample_on_device=True
        )
    assert model._vfy_penalties is not None and cls.model_capabilities["supports_spec_device_penalties"]
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = _get_prompt(130, tokenizer)[0].tolist()
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for i, (kind, penalty) in enumerate(KINDS.items()):
            runs = {}
            for path, device, drafts in (
                ("host", False, False),
                ("device", True, False),
                ("device+drafts", True, True),
            ):
                runs[path] = _Runner(cls, kv, layout, seed=10 * i + len(runs)).run(prompt, 48, penalty, device, drafts)
                out, steps, seconds = runs[path]
                logger.info(
                    f"[penalties] {kind} {path}: {len(out)} tokens in {steps} verifies, "
                    f"{1000 * seconds / steps:.1f} ms per verify, {len(out) / seconds:.1f} tokens/s"
                )
            host, device = runs["host"][0], runs["device"][0]
            assert device == host, f"{kind}: the device path left the host path at {_first_difference(device, host)}"
            drafted = runs["device+drafts"][0]
            at = _first_difference(drafted, host)
            logger.info(f"[penalties] {kind}: drafted device path equals the host path for {at} of {len(host)} tokens")
            assert at >= 16, f"{kind}: the drafted device path left the host path at token {at}"
            assert runs["device+drafts"][1] < runs["device"][1], f"{kind}: no draft was accepted"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew"
    cls.release_persistent_capture()


def _first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
