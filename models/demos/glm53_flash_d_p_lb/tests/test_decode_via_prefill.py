# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode via prefill on the LoudBox (2x4, all 45 layers), through the prefill engine's adapter / runtime API exactly
as tt-metal's prefill runner drives it (get_adapter("glm53_flash_d_p_lb"): allocate_kv_cache, build_runtime,
make_chunk_input, prefill_chunk with slot / actual_start / actual_end): every generated token is a prefill of the
chunk holding it, from that chunk's boundary (as the MiMo server and the Xing serve stack do). The KDA layers carry
recurrent state, so the runtime restores the slot's snapshot at the chunk boundary before re-running the chunk
(tt/runners/adapter.py: _position). The last real row of each chunk goes through the final norm on the device and the
LM head (fp32, host) through the runtime's hidden_sink.

  smoke:             the spec's smoke prompt, greedy; the answer must contain the expected word.
  consistency:       a prompt of 2 chunks - 3 tokens, greedy; at the steps that restore a boundary snapshot (and the
                     one that ends a chunk) the decode logits must match a cold prefill of the same tokens in another
                     slot (argmax equal, PCC >= 0.999).

  BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \\
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_decode_via_prefill.py -s

Env: GLM_DVP_CHUNK (2048), GLM_DVP_MAX_SEQ (8192), GLM_DVP_SMOKE_TOKENS (16), GLM_DVP_STEPS (6)."""

import os
import time

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
os.environ.setdefault("PREFILL_SP", str(S.mesh[0]))
os.environ.setdefault("PREFILL_TP", str(S.mesh[1]))
pytestmark = device_timeout(S)

CHUNK = int(os.environ.get("GLM_DVP_CHUNK", "2048"))
MAX_SEQ = int(os.environ.get("GLM_DVP_MAX_SEQ", "8192"))
SMOKE_TOKENS = int(os.environ.get("GLM_DVP_SMOKE_TOKENS", "16"))
STEPS = int(os.environ.get("GLM_DVP_STEPS", "6"))


class Decoder:
    """The runner's view of the model (adapter + runtime), plus last-token logits per chunk."""

    def __init__(self, mesh, num_users=2):
        from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
        from models.demos.glm53_flash_d_p.reference.weights import WeightLoader
        from models.demos.glm53_flash_d_p.tt.runners.adapter import resolve_model_path

        t0 = time.time()
        adapter = get_adapter(S.get("contract.adapter", S.model))
        hf = adapter.load_hf_config()
        n = hf.num_hidden_layers
        params = PrefillRunParams(
            mesh_shape=tuple(S.mesh),
            num_layers=n,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=MAX_SEQ,
            chunk_size=CHUNK,
            num_users=num_users,
            capacity_factor=1,
            num_links=int(S.get("contract.num_links") or 1),
            gate_mode_name=adapter.default_gate_mode,
            kv_only_last_layer=False,
            weight_cache_path=adapter.weight_cache_path(tuple(S.mesh)),
        )
        self.kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
        self.rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=params)
        self.lm_head = WeightLoader(resolve_model_path()).get("lm_head.weight").float()
        self.rt.hidden_sink = self._sink
        self.last = None
        print(f"[dvp] runtime up in {time.time() - t0:.0f} s (chunk {CHUNK}, max_seq {MAX_SEQ})", flush=True)

    def _sink(self, h, slot, start, end):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.common import replicated_to_host, split_to_host

        hidden = self.rt.model.final_norm(h)
        t = (split_to_host(hidden) if self.rt.model.layout == "split" else replicated_to_host(hidden)).float()
        ttnn.deallocate(hidden)
        rows = t.reshape(-1, t.shape[-1])
        self.last = torch.nn.functional.linear(rows[end - 1 - start], self.lm_head)

    def chunk(self, seq, slot, start, end):
        import ttnn

        inp = self.rt.make_chunk_input(seq[start:end])
        self.rt.prefill_chunk(inp, self.kv, slot_id=slot, actual_start=start, actual_end=end)
        ttnn.deallocate(inp)
        return self.last

    def prefill(self, ids, slot):
        """Cold prefill of ids from position 0 -> logits after ids[-1]."""
        logits = None
        for start in range(0, len(ids), CHUNK):
            logits = self.chunk(ids, slot, start, min(start + CHUNK, len(ids)))
        return logits

    def generate(self, ids, slot, n, stop=None):
        """Greedy: prefill, then each token re-prefills the chunk holding it. -> (tokens, logits per step, s/token)."""
        logits = self.prefill(ids, slot)
        out, steps, times = [], [logits], []
        for _ in range(n):
            t = int(logits.argmax())
            out.append(t)
            if stop is not None and stop(out):
                break
            seq = ids + out
            t0 = time.time()
            logits = self.chunk(seq, slot, (len(seq) - 1) // CHUNK * CHUNK, len(seq))
            times.append(time.time() - t0)
            steps.append(logits)
        return out, steps, times


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def smoke(decoder):
    from models.demos.common.bringup.testing.smoke import prompt_ids, tokenizer

    tok = tokenizer(S)
    ids = prompt_ids(S, tok)
    expect = S.data["intake"]["smoke"]["expect"]
    eos = tok.eos_token_id
    out, _, times = decoder.generate(
        ids, 0, SMOKE_TOKENS, stop=lambda o: o[-1] == eos or expect.lower() in tok.decode(o).lower()
    )
    text = tok.decode(out)
    st = sum(times) / max(1, len(times))
    print(f"[dvp] smoke: {S.data['intake']['smoke']['prompt']!r} -> {text!r} ({len(out)} tokens, {st:.2f} s/token)")
    return [] if expect.lower() in text.lower() else [f"smoke: expected {expect!r} in {text!r}"]


def consistency(decoder):
    from models.demos.common.bringup.testing.smoke import tokenizer

    tok = tokenizer(S)
    filler = (
        "The lighthouse keeper kept a log of every ship that passed, the weather, the tide and the colour of the sea. "
    )
    text = "Continue this logbook in the same style.\n\n" + filler * 400
    n = 2 * CHUNK - 3
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    assert len(ids) == n
    out, steps, times = decoder.generate(ids, 0, STEPS)
    print(f"[dvp] consistency: generated {tok.decode(out)!r}, {sum(times) / len(times):.2f} s/token")
    # step k: logits after ids + out[:k]; k 1 / 5 restore the boundary snapshot at C / 2 C, k 3 ends a chunk
    fails = []
    for k in (1, 3, 5):
        if k >= len(steps):
            continue
        ref = decoder.prefill(ids + out[:k], 1)
        p = _pcc(steps[k], ref)
        same = int(steps[k].argmax()) == int(ref.argmax())
        print(f"[dvp] step {k} (length {n + k}): decode vs cold prefill PCC {p:.6f}, argmax equal {same}")
        if p < 0.999 or not same:
            fails.append(f"step {k}: PCC {p:.6f}, argmax {int(steps[k].argmax())} vs {int(ref.argmax())}")
    return fails


@mesh_parametrize
def test_decode_via_prefill(mesh_device):
    decoder = Decoder(mesh_device)
    fails = smoke(decoder) + consistency(decoder)
    assert not fails, fails
