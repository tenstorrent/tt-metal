# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GLM-5.4 config, loop schedule, pass-2 attention with window carry, and the looped stack."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.sparse_sdpa_prefill import MASKED_INDEX
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import (
    LoopPass2MLAReference,
    LoopStep,
    damp_gate_reference,
    glm54_stack_reference,
    loop_schedule,
    pass1_indices,
    two_bank_sparse_mla,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4.tests.helpers import (
    pcc,
    tiny_gate_weights,
    tiny_glm54_config,
    tiny_layer_weights,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import (
    GLM54_ASSUMPTIONS,
    GLM54Config,
    check_assumptions,
    glm_5_4_hf_config,
)

# --- config ---------------------------------------------------------------------------------------------------


def test_config_holds_every_assumption():
    assert check_assumptions(glm_5_4_hf_config()) == []
    assert [a.id for a in GLM54_ASSUMPTIONS] == [
        "A1a", "A1b", "A1c", "A1d", "A1e", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
    ]  # fmt: skip


def test_check_assumptions_reports_a_mismatch():
    c = glm_5_4_hf_config()
    c.loop_window_size = 64
    del c.loop_gate_has_bias
    mismatches = check_assumptions(c)
    assert any(m.startswith("A3: loop_window_size = 64") for m in mismatches)
    assert any(m.startswith("A1d: loop_gate_has_bias = '<missing>'") for m in mismatches)


def test_config_is_glm53_plus_loop_fields():
    glm53, glm54 = vars(glm_5_3_hf_config()), vars(glm_5_4_hf_config())
    assert {k: glm54[k] for k in glm53} == glm53
    assert set(glm54) - set(glm53) == {
        "loop_first_layer", "loop_last_layer", "loop_count", "loop_window_size", "loop_window_format",
        "loop_gate_has_bias", "loop_gate_weight_dtype",
    }  # fmt: skip
    assert list(GLM54Config.loop_layers()) == list(range(26, 54))


def test_window_format_is_a_device_cache_format():
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

    assert MlaKvCacheFormat(GLM54Config.LOOP_WINDOW_FORMAT) is MlaKvCacheFormat.SCALED_FP8


def test_loop_layers_carry_seven_topk_sets():
    # A5: pass 2 reuses pass-1 top-k; the loop layers own 7 indexers, and both loop and coda start full.
    types = GLM54Config.indexer_types()
    loop = [types[i] for i in GLM54Config.loop_layers()]
    assert loop.count("full") == 7 and loop[0] == "full"
    assert types[GLM54Config.LOOP_LAST_LAYER + 1] == "full"


# --- schedule -------------------------------------------------------------------------------------------------


def test_schedule_small():
    L, G = LoopStep, lambda k: LoopStep("gate", k)
    assert loop_schedule(4, 1, 2, 2) == [
        L("layer", 0, 0), L("layer", 1, 1), L("layer", 1, 2), G(1), L("layer", 2, 1), L("layer", 2, 2), G(2),
        L("layer", 0, 3),
    ]  # fmt: skip


def test_schedule_glm54_runs_106_layer_passes():
    steps = loop_schedule(78, 26, 53, 2)
    layers = [s for s in steps if s.kind == "layer"]
    assert len(layers) == 106 and [s.loop_pass for s in steps if s.kind == "gate"] == [1, 2]
    assert [s.layer for s in layers[26:54]] == [s.layer for s in layers[54:82]] == list(range(26, 54))
    assert steps[54] == LoopStep("gate", 1) and steps[83] == LoopStep("gate", 2) and layers[-1].layer == 77


# --- pass-2 attention -----------------------------------------------------------------------------------------


def _mla_ref(config, seq_len, layer=1):
    return SparseMLAReference(config, tiny_layer_weights(config, layer)["mla_weights"], seq_len=seq_len)


def test_pass2_projections_are_pass1_projections():
    # A2/A4: pass 2 projects with the pass-1 layer's weights at the pass-1 positions, so the same input gives
    # the pass-1 KVPE rows. Chunked (start 8) checks the positions follow actual_start.
    seq = 16
    config = tiny_glm54_config(seq_len=seq)
    ref = _mla_ref(config, seq)
    x = torch.randn(1, seq, config.hidden_size).to(torch.bfloat16)
    ref.forward(x[:, :8])
    ref.forward(x[:, 8:], actual_start=8)
    _, kvpe = ref.absorbed_query_and_kvpe(x[:, 8:], actual_start=8)
    assert torch.equal(kvpe, ref.kvpe_cache[0, 0, 8:])


def test_pass2_tail_reproduces_pass1_attention():
    # With bank B masked and bank A = the pass-1 cache at the pass-1 top-k, the pass-2 path (absorbed query,
    # two-bank op, latent output) is the pass-1 attention. Ties the shared projection/output path to GLM-5.3.
    seq = 16
    config = tiny_glm54_config(seq_len=seq)
    ref = _mla_ref(config, seq)
    x = torch.randn(1, seq, config.hidden_size).to(torch.bfloat16)
    expected = ref.forward(x)
    q, _ = ref.absorbed_query_and_kvpe(x)
    idx_a = pass1_indices(ref.indexer_topk, 0, config.index_topk)
    idx_b = torch.full((1, 1, seq, 1), MASKED_INDEX, dtype=torch.int64)
    kvpe_a = ref.kvpe_cache[0, 0]
    latent = two_bank_sparse_mla(q, kvpe_a, idx_a, kvpe_a[:1], idx_b, ref.softmax_scale, latent_dim=config.kv_lora_rank)
    got = ref.latent_attention_output(latent, x)
    assert pcc(got, expected) > 0.999


@pytest.mark.parametrize("window, chunk", [(4, 5), (8, 3)], ids=["window_lt_chunk", "window_spans_chunks"])
def test_pass2_window_carry_chunked_equals_single_shot(window, chunk, expect_error):
    seq = 15
    config = tiny_glm54_config(seq_len=seq, window=window)
    ref = _mla_ref(config, seq)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, seq, config.hidden_size, generator=g).to(torch.bfloat16)
    kvpe_a = torch.randn(seq, config.kv_lora_rank + config.qk_rope_head_dim, generator=g).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(seq, generator=g)[: config.index_topk] for _ in range(seq)]).unsqueeze(0)

    single = LoopPass2MLAReference(ref, window=window, latent_dim=config.kv_lora_rank)
    want = single.forward(x, kvpe_a=kvpe_a, indices_a=pass1_indices(topk, 0, config.index_topk))

    chunked = LoopPass2MLAReference(ref, window=window, latent_dim=config.kv_lora_rank)
    got, kvpe_rows = [], []
    for s in range(0, seq, chunk):
        idx_a = pass1_indices(topk[:, s : s + chunk], s, config.index_topk)
        got.append(chunked.forward(x[:, s : s + chunk], kvpe_a=kvpe_a, indices_a=idx_a, actual_start=s))
        kvpe_rows.append(chunked.last_kvpe)
        # The carry is the last window-1 pass-2 KVPE rows seen so far.
        seen = torch.cat(kvpe_rows)
        assert torch.equal(chunked.window_carry, seen[seen.shape[0] - min(window - 1, seen.shape[0]) :])
    torch.testing.assert_close(torch.cat(got, dim=1), want, rtol=0, atol=0)
    assert torch.equal(torch.cat(kvpe_rows), single.last_kvpe)

    with expect_error(ValueError, "expected 15"):
        chunked.forward(x[:, :chunk], kvpe_a=kvpe_a, indices_a=pass1_indices(topk[:, :chunk], 0, 8), actual_start=0)


# --- looped stack ---------------------------------------------------------------------------------------------


def _stack(seq, chunk, per_channel=False, index_topk=8):
    config = tiny_glm54_config(seq_len=seq, index_topk=index_topk)
    layers = [tiny_layer_weights(config, i) for i in range(config.num_hidden_layers)]
    gates = [tiny_gate_weights(config, k, per_channel=per_channel) for k in (1, 2)]
    x = torch.randn(1, seq, config.hidden_size, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    return config, gates, glm54_stack_reference(config, layers, gates, x, chunk_size=chunk)


@pytest.mark.parametrize("per_channel", [False, True], ids=["scalar_gate", "channel_gate"])
def test_looped_stack_runs_and_wires(per_channel):
    # Reduced config: L0 prelude, L1-L2 loop x2 + gates, L3 coda; 24 tokens in chunks of 8, window 4, top-8.
    config, gates, result = _stack(24, 8, per_channel)
    cp = result.checkpoints
    assert result.output.shape == (1, 24, config.hidden_size) and torch.isfinite(result.output).all()

    # Gate k reads pass k's input and output, and its output is pass k+1's input (A1c).
    for k in (1, 2):
        g = gates[k - 1]
        want, beta = damp_gate_reference(
            cp[f"pass{k}.in"], cp[f"pass{k}.out"], norm_weight=g["norm_weight"], proj_weight=g["proj_weight"],
            proj_bias=g["proj_bias"], eps=config.rms_norm_eps,
        )  # fmt: skip
        assert torch.equal(cp[f"gate{k}.out"], want.to(torch.bfloat16))
        assert beta.shape[-1] == (config.hidden_size if per_channel else 1)
    assert torch.equal(cp["pass2.in"], cp["gate1.out"])
    assert not torch.equal(cp["gate1.out"], cp["pass1.out"])

    # The shared loop layer L2 reuses L1's pass-1 top-k; sentinels mark the short early rows (A5).
    assert torch.equal(cp["L2.pass1.topk"], cp["L1.pass1.topk"])
    assert cp["L1.pass1.topk"].shape == (1, 1, 24, config.index_topk)
    assert (cp["L1.pass1.topk"][0, 0, 0] == MASKED_INDEX).sum() == config.index_topk - 1
    # Pass 2 writes its own latents (different input), not the pass-1 cache.
    assert cp["L1.pass2.kvpe"].shape == cp["L1.pass1.kvpe"].shape
    assert pcc(cp["L1.pass2.kvpe"], cp["L1.pass1.kvpe"]) < 0.999


def test_looped_stack_sparse_chunked_first_chunk_exact_and_deterministic():
    # Top-8 of 24: the first chunk sees the same inputs either way, so every checkpoint matches bit for bit.
    # Later chunks differ: single-shot pass 1 runs the MHA branch and chunked the absorbed branch, and their bf16
    # rounding flips near-tied indexer top-k picks (stack PCC ~0.997 on these synthetic weights); the dense
    # selection test below bounds the remaining rounding.
    _, _, single = _stack(24, None)
    _, _, chunked = _stack(24, 8)
    _, _, again = _stack(24, 8)
    assert torch.equal(chunked.output, again.output)
    assert all(torch.equal(chunked.checkpoints[n], again.checkpoints[n]) for n in chunked.checkpoints)
    assert torch.equal(chunked.output[:, :8], single.output[:, :8])
    for name, t in single.checkpoints.items():
        got, want = chunked.checkpoints[name][..., :8, :], t[..., :8, :]
        if name.endswith("topk"):  # a top-k row is a set; its order depends on the scored width
            got, want = got.sort(-1).values, want.sort(-1).values
        assert torch.equal(got, want), name


def test_looped_stack_dense_selection_chunked_matches_single_shot():
    # index_topk >= seq selects every causal key, so no selection can flip: chunked differs from single-shot by
    # bf16 branch rounding only (measured 0.99987 at the output).
    _, _, single = _stack(24, None, index_topk=24)
    _, _, chunked = _stack(24, 8, index_topk=24)
    assert pcc(chunked.output, single.output) > 0.999
    for name in ("gate1.out", "L1.pass2.kvpe", "L2.pass2.kvpe", "gate2.out"):
        assert pcc(chunked.checkpoints[name], single.checkpoints[name]) > 0.999, name
