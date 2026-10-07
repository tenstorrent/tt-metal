# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the e2e harness; no device needed."""
from pathlib import Path

import pytest
import torch

from models.experimental.ops.quasar.qwen3_vl.tests.e2e import pcc as P
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID, RunConfig, parse_grid
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import PRESETS, build_inputs
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.progress import ProgressLog
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import StageRecorder
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import truncate_hf_config, vision_padded_seq_len


def _opts(**over):
    base = {
        "--qwen-size": "tiny",
        "--qwen-vision-layers": 2,
        "--qwen-text-layers": 2,
        "--qwen-decode-steps": 1,
        "--qwen-deepstack-at": None,
        "--qwen-kv-blocks": None,
        "--qwen-host-ops": "",
        "--qwen-disable-wa": "",
        "--qwen-allow-uncertified": False,
        "--qwen-quasar-config": False,
        "--qwen-expect-grid": None,
        "--qwen-run-dir": "/tmp/x",
    }
    base.update(over)
    return base.__getitem__


def test_run_config_defaults():
    cfg = RunConfig.from_options(_opts())
    assert (cfg.vision_layers, cfg.text_layers, cfg.decode_steps) == (2, 2, 1)
    assert cfg.deepstack_at is None and cfg.host_ops == () and cfg.run_dir == Path("/tmp/x")


def test_run_config_lists_and_grid():
    cfg = RunConfig.from_options(
        _opts(**{"--qwen-host-ops": "linear, rms_norm", "--qwen-expect-grid": "3x2", "--qwen-deepstack-at": 0})
    )
    assert cfg.host_ops == ("linear", "rms_norm")
    assert cfg.expect_grid == (3, 2) and cfg.deepstack_at == 0


def test_parse_grid_rejects_garbage(expect_error):
    with expect_error(ValueError, "grid must look like"):
        parse_grid("3by2")


def test_cache_key_covers_everything():
    a = RunConfig.from_options(_opts())
    keys = {
        a.cache_key((3, 2)),
        a.cache_key((8, 4)),
        RunConfig.from_options(_opts(**{"--qwen-text-layers": 3})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-deepstack-at": 0})).cache_key((3, 2)),
        RunConfig.from_options(_opts(**{"--qwen-quasar-config": True})).cache_key((3, 2)),
    }
    assert len(keys) == 5


@pytest.mark.parametrize(
    "name, grid, image_tokens, seq",
    [("tiny", [1, 16, 16], 64, 78), ("demo", [1, 86, 128], 2752, 2766)],
)
def test_preset_token_counts(name, grid, image_tokens, seq):
    from transformers import AutoProcessor

    inputs = build_inputs(PRESETS[name], AutoProcessor.from_pretrained(HF_MODEL_ID))
    assert inputs["image_grid_thw"][0].tolist() == grid
    assert int(inputs["image_grid_thw"][0].prod()) // 4 == image_tokens
    assert inputs["input_ids"].shape[1] == seq


def test_preset_kv_capacity_covers_seq():
    for p in PRESETS.values():
        assert p.kv_blocks * p.block_size >= p.max_seq_len


@pytest.mark.parametrize(
    "n, want",
    [
        (1, 128),
        (216, 256),
        (256, 256),
        (1024, 1024),
        (1025, 2048),
        (1152, 2048),
        (2048, 2048),
        (2049, 4096),
        (11008, 12288),
    ],
)
def test_vision_padded_seq_len(n, want):
    assert vision_padded_seq_len(n) == want


def test_truncate_hf_config_same_for_tt_and_hf():
    from transformers import AutoConfig

    a = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, 0)
    b = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, 0)
    assert a.vision_config.depth == b.vision_config.depth == 2
    assert a.text_config.num_hidden_layers == 2
    assert a.vision_config.deepstack_visual_indexes == b.vision_config.deepstack_visual_indexes == [0]


def test_truncate_hf_config_keeps_real_taps():
    from transformers import AutoConfig

    c = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), 2, 2, None)
    assert c.vision_config.deepstack_visual_indexes == [5, 11, 17]


def test_reference_tiny_stage_shapes():
    from transformers import AutoProcessor

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e.host_reference import load_hf_model, run_reference

    inputs = build_inputs(PRESETS["tiny"], AutoProcessor.from_pretrained(HF_MODEL_ID))
    g = run_reference(load_hf_model(2, 2, 0), inputs, decode_steps=2)
    assert g.prefill_len == 78 and g.num_patches == 256 and len(g.teacher_tokens) == 2
    assert g.tensors["vision.block1"].shape == (256, 1024)
    assert g.tensors["vision.deepstack0"].shape == (64, 2560)
    assert g.tensors["vision.merger"].shape == (64, 2560)
    assert g.tensors["text.layer1"].shape == (78, 2560)
    assert g.tensors["text.norm"].shape == (2560,)
    assert g.tensors["text.logits.decode1"].shape == (151936,)
    assert set(g.tensors) == {
        "vision.block0",
        "vision.block1",
        "vision.deepstack0",
        "vision.merger",
        "text.layer0",
        "text.layer1",
        "text.norm",
        "text.logits.prefill",
        "text.logits.decode0",
        "text.logits.decode1",
    } | {f"vision.block{i}.{s}" for i in (0, 1) for s in ("attn", "mlp")} | {
        f"text.layer{i}.{s}" for i in (0, 1) for s in ("attn", "mlp")
    }
    assert g.tensors["vision.block0.attn"].shape == (256, 1024) and g.tensors["text.layer1.mlp"].shape == (78, 2560)


def test_pcc_identical_and_constant():
    x = torch.randn(64, 32)
    assert P.pcc(x, x) == pytest.approx(1.0)
    assert P.pcc(torch.ones(4), torch.ones(4)) == 1.0
    assert P.pcc(torch.ones(4), torch.zeros(4)) == 0.0


def test_compare_statuses():
    g = {"a": torch.randn(10, 4), "b": torch.randn(10, 4), "c": torch.randn(3), "d": torch.randn(3)}
    act = {
        "a": g["a"].clone(),
        "b": torch.randn(10, 4),
        "c": torch.tensor([1.0, float("nan"), 0.0]),
        "d": torch.randn(4),
    }
    res = {r.stage: r.status for r in P.compare(g, act, lambda s: 0.99, {})}
    assert res == {"a": "PASS", "b": "FAIL", "c": "NONFINITE", "d": "SHAPE"}
    act.pop("a")
    assert {r.stage: r.status for r in P.compare(g, act, lambda s: 0.99, {})}["a"] == "MISSING"


def test_verdict_first_failure_in_stage_order():
    rs = [
        P.StageResult("vision.block0", 1, 0, 0.99, "PASS", 0),
        P.StageResult("text.layer0", 0.5, 1, 0.99, "FAIL", 0),
        P.StageResult("text.layer1", 0.4, 1, 0.99, "FAIL", 0),
    ]
    v = P.verdict(rs, [], {}, [])
    assert v.status == "FAIL" and v.first_failure == "text.layer0" and "text.layer0" in v.markdown


def test_verdict_host_ops_is_diagnostic_even_when_all_pass():
    rs = [P.StageResult("vision.block0", 1, 0, 0.99, "PASS", 0)]
    v = P.verdict(rs, ["ttnn.linear"], {"host:ttnn.linear": 3}, [])
    assert v.status == "DIAGNOSTIC" and "ttnn.linear" in v.markdown


def test_thresholds_patterns_and_preset_override(tmp_path, expect_error):
    f = tmp_path / "t.json"
    f.write_text('{"default": {"text.layer*": 0.99, "text.logits.*": 0.98}, "demo": {"text.layer*": 0.97}}')
    assert P.thresholds_for("tiny", f)("text.layer1") == 0.99
    assert P.thresholds_for("demo", f)("text.layer1") == 0.97
    assert P.thresholds_for("tiny", f)("text.logits.decode0") == 0.98
    with expect_error(KeyError, "no threshold"):
        P.thresholds_for("tiny", f)("unknown.stage")


def test_shipped_thresholds_cover_all_stages():
    t = P.thresholds_for("tiny")
    for s in [
        "vision.block0",
        "vision.deepstack0",
        "vision.merger",
        "text.layer3",
        "text.norm",
        "text.logits.prefill",
        "text.logits.decode2",
    ]:
        assert 0.9 < t(s) <= 1.0


class _Op:
    python_fully_qualified_name = "ttnn.fake"


def test_progress_log_unfinished(tmp_path):
    log = ProgressLog(tmp_path / "progress.log")
    log.stage = "text.layer0"
    log.pre(_Op(), (torch.zeros(2, 3),), {})
    log.post(_Op(), (torch.zeros(2, 3),), {}, None)
    log.pre(_Op(), (torch.zeros(4),), {"bias": torch.zeros(7), "dtype": 1})
    assert log.last_unfinished().startswith("ttnn.fake")
    assert "bias=(shape=[7]" in log.last_unfinished() and "dtype=1" not in log.last_unfinished()
    text = (tmp_path / "progress.log").read_text()
    assert "PRE ttnn.fake stage=text.layer0" in text and "POST ttnn.fake" in text


def test_recorder_wrap_transform_append(tmp_path, monkeypatch):
    class Mod:
        def forward(self, x, mode="prefill"):
            return x * 2

    m = Mod()
    rec = StageRecorder(ProgressLog(tmp_path / "p.log"), to_host=lambda t: t.float())
    rec.wrap(
        monkeypatch,
        m,
        "text.layer0",
        lambda t: t[:3],
        when=lambda a, k: k.get("mode", "prefill") == "prefill",
        append_dim=0,
    )
    m.forward(torch.ones(4, 2))
    m.forward(torch.ones(4, 2), mode="decode")
    m.forward(torch.ones(4, 2))
    assert rec.tensors["text.layer0"].shape == (6, 2)
    assert torch.equal(rec.tensors["text.layer0"], torch.full((6, 2), 2.0))
    assert rec.seconds["text.layer0"] >= 0


def test_bf16_precision_everywhere():
    import ttnn
    from models.tt_transformers.tt.model_config import TensorGroup

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import bf16_decoders_precision

    p = bf16_decoders_precision(2, "Qwen3-VL-4B-Instruct")
    for d in range(2):
        for g in TensorGroup:
            assert p.get_tensor_dtype(d, g) == ttnn.bfloat16, g
    conf = p.decoder_optimizations[0]
    # HiFi4 without fp32 dest accumulation: fp32 partials reloaded through srcA are undefined (ttsim, Quasar llama).
    assert all(v.value == "hifi4fp16" for v in conf.op_fidelity_settings.values())


def test_model_args_classes_selection():
    from models.experimental.ops.quasar.qwen3_vl.tt.model_config import VisionModelArgs
    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import (
        QuasarModelArgs,
        QuasarVisionModelArgs,
        model_args_classes,
    )
    from models.tt_transformers.tt.model_config import ModelArgs

    assert model_args_classes(force=True) == (QuasarModelArgs, QuasarVisionModelArgs)
    assert issubclass(QuasarVisionModelArgs, VisionModelArgs) and issubclass(QuasarModelArgs, ModelArgs)


def test_resolve_dotted_target():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    parent, attr = O.resolve("ttnn.experimental.paged_update_cache")
    assert parent is ttnn.experimental and attr == "paged_update_cache"


def test_fp32_host_cast_workaround_predicate_and_rewrite():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "host_cast_fp32_upload")
    x = torch.randn(4, 4)
    assert wa.applies((x,), {"dtype": ttnn.bfloat16, "device": object()})
    assert not wa.applies((x,), {"dtype": ttnn.bfloat16})  # host-only conversion is fine
    assert not wa.applies((x,), {"dtype": ttnn.float32, "device": object()})
    assert not wa.applies((x.bfloat16(),), {"dtype": ttnn.bfloat16, "device": object()})
    seen = {}
    wa.rewrite(lambda t, **kw: seen.update(dtype=t.dtype), (x,), {"dtype": ttnn.bfloat16, "device": object()})
    assert seen["dtype"] == torch.bfloat16


def test_session_installs_and_counts_workarounds(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    calls = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: calls.append(t.dtype))
    s = O.OverrideSession(mesh_device=None, host_ops=(), disable_wa=())
    s.install(monkeypatch)
    ttnn.from_torch(torch.randn(2, 2), dtype=ttnn.bfloat16, device=object())
    ttnn.from_torch(torch.randn(2, 2), dtype=ttnn.bfloat16)
    assert calls == [torch.bfloat16, torch.float32]
    assert s.hits["wa:host_cast_fp32_upload"] == 1


def test_session_disable_workaround(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    calls = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: calls.append(t.dtype))
    O.OverrideSession(None, (), ("host_cast_fp32_upload",)).install(monkeypatch)
    ttnn.from_torch(torch.randn(2, 2), dtype=ttnn.bfloat16, device=object())
    assert calls == [torch.float32]


def test_strip_fp32_dest_acc_rewrites_only_fp32_configs():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import strip_fp32_dest_acc

    class Args:
        pass

    a = Args()
    a.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    keep = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    a.compute_kernel_config_hifi2_fp16 = keep
    a.unrelated = 3
    assert strip_fp32_dest_acc(a) == ["compute_kernel_config_hifi4"]
    c = a.compute_kernel_config_hifi4
    assert not c.fp32_dest_acc_en and c.packer_l1_acc and c.math_fidelity == ttnn.MathFidelity.HiFi4
    assert a.compute_kernel_config_hifi2_fp16 is keep


def test_scatter_rows_matches_index_put():
    from models.experimental.ops.quasar.qwen3_vl.tt.common import scatter_rows

    base = torch.randn(10, 4)
    rows = torch.randn(3, 4)
    idx = torch.tensor([2, 3, 7])
    out = scatter_rows(base, idx, rows)
    want = base.clone()
    want[idx] = rows
    assert torch.equal(out, want) and not torch.equal(base, want)  # input left untouched


def test_bf16_precision_keeps_fp32_dest_acc_when_asked():
    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import bf16_decoders_precision

    conf = bf16_decoders_precision(1, "Qwen3-VL-4B-Instruct", fp32_dest_acc=True).decoder_optimizations[0]
    assert all(v.value == "hifi4" for v in conf.op_fidelity_settings.values())


def test_fp32_dest_acc_requested_reads_env(monkeypatch):
    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fp32_dest_acc_requested

    monkeypatch.delenv("QWEN_QSR_FP32_DEST_ACC", raising=False)
    assert not fp32_dest_acc_requested()
    monkeypatch.setenv("QWEN_QSR_FP32_DEST_ACC", "1")
    assert fp32_dest_acc_requested()


def _bounded_args(x, y):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import _QuasarArgsMixin

    a = _QuasarArgsMixin.__new__(_QuasarArgsMixin)
    a.max_grid_size = ttnn.CoreGrid(x=x, y=y)
    return a


@pytest.mark.parametrize("x, y", [(2, 1), (8, 4), (8, 8)])
def test_grid_helpers_stay_inside_device_grid(x, y):
    a = _bounded_args(x, y)
    for n in (80, 304, 192, 7):  # dim, hidden, qkv tiles, and a prime
        rows, cols = a.find_grid(n)
        assert rows <= y and cols <= x and n % (rows * cols) == 0
        gx, gy = a.find_prefill_grid(n, n)  # consumers read it as (x, y)
        assert gx <= x and gy <= y
    rows, cols = a.find_grid_k_n(80, 304)
    assert rows <= y and cols <= x and 80 % (rows * cols) == 0 and 304 % (rows * cols) == 0


def test_grid_helpers_match_base_on_full_wh_grid(monkeypatch):
    from models.tt_transformers.tt import model_config as mc
    from models.tt_transformers.tt.model_config import ModelArgs

    monkeypatch.setattr(mc, "is_wormhole_b0", lambda *a, **k: True)  # base find_grid picks WH's 8x8 bounds

    a = _bounded_args(8, 8)
    for n in (80, 304, 192):
        assert a.find_grid(n) == ModelArgs.find_grid(a, n)
        assert a.find_prefill_grid(n, n) == ModelArgs.find_prefill_grid(a, n, n)
    assert a.find_grid_k_n(80, 304) == ModelArgs.find_grid_k_n(a, 80, 304)


def test_find_prefill_grid_is_x_then_y():
    a = _bounded_args(2, 1)
    assert a.find_prefill_grid(4, 80) == (2, 1)  # x divides the column tiles, y the row tiles


def test_fit_matmul_config_rescales_per_core_work():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fit_matmul_config

    c = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=24,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    f = fit_matmul_config(c, 2, 1)
    g = f.compute_with_storage_grid_size
    assert (g.x, g.y) == (2, 1)
    assert f.per_core_M == 4 * 8 and f.per_core_N == 24 * 4  # same total M x N work spread over fewer cores
    assert f.fuse_batch and f.in0_block_w == 1
    assert fit_matmul_config(c, 8, 8) is c  # fits already: untouched

    m = ttnn.MinimalMatmulConfig(
        M_block_size=8, K_block_size=8, N_block_size=8, compute_with_storage_grid_size=ttnn.CoreCoord(8, 8)
    )
    fm = fit_matmul_config(m, 2, 1)
    assert (fm.compute_with_storage_grid_size.x, fm.compute_with_storage_grid_size.y) == (2, 1)
    assert (fm.M_block_size, fm.K_block_size, fm.N_block_size) == (8, 8, 8)
    assert fit_matmul_config(None, 2, 1) is None


def test_fit_matmul_config_uses_real_tile_counts():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fit_matmul_config

    c = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=24,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    f = fit_matmul_config(c, 2, 1, m_tiles=4, n_tiles=192)  # seq 128 x qkv 6144
    assert (f.per_core_M, f.per_core_N) == (4, 96)


def test_fit_matmul_config_minimizes_l1_when_shrinking():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fit_matmul_config

    c = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    assert fit_matmul_config(c, 2, 1, m_tiles=4, n_tiles=80).in0_block_w == 1  # fewer cores: smallest K block


def test_cache_key_tracks_weight_layout_version(monkeypatch):
    from models.experimental.ops.quasar.qwen3_vl.tt import quasar_config as qc

    cfg = RunConfig.from_options(_opts(**{"--qwen-quasar-config": True}))
    a = cfg.cache_key((8, 8))
    monkeypatch.setattr(qc, "WEIGHT_LAYOUT_VERSION", qc.WEIGHT_LAYOUT_VERSION + 1)
    assert cfg.cache_key((8, 8)) != a  # cached tensors keep their memory config, so a layout change needs a new key


def test_stage_order_puts_sublayers_after_their_block():
    stages = ["text.layer0", "vision.block1", "vision.block0.mlp", "vision.block0", "vision.block0.attn"]
    assert sorted(stages, key=P._stage_sort_key) == [
        "vision.block0.attn",
        "vision.block0.mlp",
        "vision.block0",
        "vision.block1",
        "text.layer0",
    ]


def test_fit_matmul_config_force_rebuilds_a_config_that_already_fits():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fit_matmul_config

    c = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=4,
        per_core_N=40,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    assert fit_matmul_config(c, 2, 1) is c
    f = fit_matmul_config(c, 2, 1, m_tiles=4, n_tiles=80, force=True)
    assert f.in0_block_w == 1 and (f.per_core_M, f.per_core_N) == (4, 40)


def test_fit_matmul_config_blocks_large_per_core_outputs():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import fit_matmul_config

    c = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=38,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    f = fit_matmul_config(c, 2, 1, m_tiles=32, n_tiles=304)  # demo-size MLP up-projection on 2 cores
    assert (f.per_core_M, f.per_core_N) == (32, 152)
    assert (f.out_block_h, f.out_block_w) == (8, 19)  # largest divisors <= 8 rows / 32 columns of tiles


class _FakeGrid:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeDev:
    def __init__(self, x, y):
        self._g = _FakeGrid(x, y)

    def compute_with_storage_grid_size(self):
        return self._g

    def get_num_devices(self):
        return 1


class _FakeTensor:
    def __init__(self, x, y):
        self._d = _FakeDev(x, y)

    def device(self):
        return self._d


def test_concat_heads_decode_workaround_only_on_grids_smaller_than_heads():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "concat_heads_decode_small_grid")
    assert wa.target == "ttnn.experimental.nlp_concat_heads_decode"
    assert wa.applies((_FakeTensor(2, 1),), {"num_heads": 32})
    assert not wa.applies((_FakeTensor(8, 8),), {"num_heads": 32})
    assert not wa.applies((_FakeTensor(2, 1),), {})  # no head count: leave the op alone


class _FakeShardTensor(_FakeTensor):
    def __init__(self, x, y, sharded):
        super().__init__(x, y)
        self._sharded = sharded

    def is_sharded(self):
        return self._sharded


class _FakeOut:
    def __init__(self, shape, x, y):
        import ttnn

        self.padded_shape = ttnn.Shape(shape)
        self._grid = ttnn.CoreCoord(x, y)

    def device(self):
        return self

    def compute_with_storage_grid_size(self):
        return self._grid


def test_with_shard_spec_completes_generic_sharded_configs():
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    width = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    spec = O._with_shard_spec(_FakeOut([1, 1, 32, 2560], 2, 1), width).shard_spec
    assert list(spec.shape) == [32, 1280] and spec.grid.num_cores() == 2
    spec = O._with_shard_spec(_FakeOut([1, 1, 32, 96], 2, 1), width).shard_spec  # 3 tiles: only 1 core divides
    assert list(spec.shape) == [32, 96] and spec.grid.num_cores() == 1
    height = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    spec = O._with_shard_spec(_FakeOut([1, 2, 64, 128], 2, 1), height).shard_spec
    assert list(spec.shape) == [64, 128] and spec.grid.num_cores() == 2


class _FakeRowTensor(_FakeShardTensor):
    def __init__(self, x, y, width, sharded=False):
        super().__init__(x, y, sharded)
        self.padded_shape = [1, 1, 32, width]


def test_untilize_single_core_workaround_predicate():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "small_grid_untilize_single_core")
    assert wa.target == "ttnn.untilize"
    assert wa.applies((_FakeRowTensor(2, 1, 151936),), {"use_multicore": True})
    assert not wa.applies((_FakeRowTensor(8, 8, 151936),), {"use_multicore": True})  # full grid: stock
    assert not wa.applies((_FakeRowTensor(2, 1, 128),), {"use_multicore": True})  # narrow rows fit
    assert not wa.applies((_FakeRowTensor(2, 1, 151936, sharded=True),), {})  # sharded factories differ
    assert not wa.applies((_FakeRowTensor(2, 1, 151936),), {"use_multicore": False})  # already single core


def test_unshard_linear_workaround_predicate():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "small_grid_unshard_linear")
    assert wa.target == "ttnn.linear"
    w = object()
    assert wa.applies((_FakeShardTensor(2, 1, True), w), {})
    assert not wa.applies((_FakeShardTensor(8, 8, True), w), {})  # full grid: stock behaviour
    assert not wa.applies((_FakeShardTensor(2, 1, False), w), {})  # interleaved input: nothing to do
    assert not wa.applies((_FakeShardTensor(2, 1, True), w), {"program_config": object()})  # explicit config wins


def test_uncertified_fallback_refused(monkeypatch, expect_error):
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    monkeypatch.setattr(O, "CERTIFIED", {})
    s = O.OverrideSession(mesh_device=None, host_ops=("linear",), disable_wa=())
    with expect_error(RuntimeError, "not certified"):
        s.install(monkeypatch)
    assert s.host_ops_active == []


def test_unknown_host_op_rejected(monkeypatch, expect_error):
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    s = O.OverrideSession(mesh_device=None, host_ops=("no_such_op",), disable_wa=())
    with expect_error(KeyError, "no_such_op"):
        s.install(monkeypatch)


def test_host_op_short_and_full_names_and_all():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    pick = lambda names: [f.target for f in O.OverrideSession(None, names, ())._selected_fallbacks()]
    assert pick(("linear",)) == ["ttnn.linear"]
    assert pick(("ttnn.experimental.paged_fill_cache",)) == ["ttnn.experimental.paged_fill_cache"]
    assert sorted(pick(("all",))) == sorted(O.FALLBACKS)


def test_allowed_uncertified_fallback_installs_and_counts(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    monkeypatch.setattr(O, "CERTIFIED", {})
    monkeypatch.setattr(O, "WORKAROUNDS", [])
    original = ttnn.linear
    s = O.OverrideSession(mesh_device=None, host_ops=("linear",), disable_wa=(), allow_uncertified=True)
    s.install(monkeypatch)
    assert ttnn.linear is not original and s.host_ops_active == ["ttnn.linear"]


def test_hand_rope_matches_rotate_half_definition():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    x = torch.randn(1, 2, 64, 128)
    cos, sin = torch.randn(1, 1, 64, 128), torch.randn(1, 1, 64, 128)
    t = torch.zeros(32, 32)
    for i in range(0, 32, 2):
        t[i, i + 1], t[i + 1, i] = 1.0, -1.0
    out = O.FALLBACKS["ttnn.experimental.rotary_embedding_llama"].torch_fn([x, cos, sin, t.reshape(1, 1, 32, 32)], {})
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x.shape)
    assert torch.allclose(out, x * cos + rot * sin, atol=1e-5)


def test_hand_paged_update_cache():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    cache = torch.zeros(4, 2, 32, 8)
    upd = torch.randn(1, 1, 32, 8)  # [1, batch, kv_heads padded to a tile, head_dim]
    page_table = torch.tensor([[2, 0, 1, 3]])
    O.FALLBACKS["ttnn.experimental.paged_update_cache"].torch_fn(
        [cache, upd], {"update_idxs_tensor": torch.tensor([33]), "page_table": page_table}
    )
    # Position 33 is logical block 1 -> physical block page_table[0, 1] = 0, row 33 % 32 = 1.
    assert torch.equal(cache[0, :, 1, :], upd[0, 0, :2, :])
    assert cache.abs().sum() == upd[0, 0, :2, :].abs().sum()


def test_hand_paged_fill_cache():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    cache = torch.zeros(4, 2, 32, 8)
    x = torch.randn(1, 2, 64, 8)
    page_table = torch.tensor([[3, 1, 0, 2]])
    O.FALLBACKS["ttnn.experimental.paged_fill_cache"].torch_fn([cache, x, page_table], {"batch_idx": 0})
    assert torch.equal(cache[3], x[0, :, :32]) and torch.equal(cache[1], x[0, :, 32:])
    assert cache[0].abs().sum() == 0 and cache[2].abs().sum() == 0


def test_golden_fallback_still_works_after_wrappers_installed(monkeypatch):
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    monkeypatch.setattr(O, "_GOLDENS", {})
    s = O.OverrideSession(mesh_device=None, host_ops=("linear",), disable_wa=(), allow_uncertified=True)
    s.install(monkeypatch)  # ttnn.linear is now wrapped twice (workaround, then fallback)
    a, b = torch.randn(32, 64), torch.randn(64, 16)
    assert torch.allclose(O.FALLBACKS["ttnn.linear"].torch_fn([a, b], {}).float(), a @ b, atol=1e-4)


def test_every_fallback_is_certified():
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    assert set(O.CERTIFIED) == set(O.FALLBACKS)  # so --host-ops all needs no --qwen-allow-uncertified


class _ArchTensor:
    def __init__(self, arch):
        self._arch = arch

    def device(self):
        return self

    def arch(self):
        return self._arch


def test_quasar_experimental_add_routes_only_on_quasar(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "quasar_experimental_add")
    assert wa.target == "ttnn.add"
    monkeypatch.setattr(ttnn, "Tensor", _ArchTensor)  # the predicate looks for ttnn.Tensor arguments
    assert wa.applies((_ArchTensor(ttnn.device.Arch.QUASAR), 1.0), {})
    assert not wa.applies((_ArchTensor(ttnn.device.Arch.WORMHOLE_B0), 1.0), {})
    assert not wa.applies((1.0, 2.0), {})  # no tensor argument: leave the op alone
    calls = []
    monkeypatch.setattr(ttnn.experimental.quasar, "add", lambda *a, **k: calls.append((a, k)) or "q")
    assert wa.rewrite(None, ("a", "b"), {"memory_config": "m"}) == "q" and calls == [
        (("a", "b"), {"memory_config": "m"})
    ]


def test_quasar_experimental_sdpa_resolves_dotted_name(monkeypatch):
    import types

    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "quasar_experimental_sdpa")
    assert wa.target == "ttnn.transformer.scaled_dot_product_attention"
    fake = types.SimpleNamespace(scaled_dot_product_attention=lambda *a, **k: ("q-sdpa", a, k))
    monkeypatch.setattr(ttnn.experimental.quasar, "transformer", fake)
    assert wa.rewrite(None, ("q", "k", "v"), {"is_causal": False}) == ("q-sdpa", ("q", "k", "v"), {"is_causal": False})


def test_quasar_rms_norm_bf16_dest_only_rewrites_fp32_configs_on_quasar(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    wa = next(w for w in O.WORKAROUNDS if w.name == "quasar_rms_norm_bf16_dest")
    assert wa.target == "ttnn.rms_norm"
    fp32 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    bf16 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    monkeypatch.setattr(ttnn, "Tensor", _ArchTensor)
    q, wh = _ArchTensor(ttnn.device.Arch.QUASAR), _ArchTensor(ttnn.device.Arch.WORMHOLE_B0)
    assert wa.applies((q,), {"compute_kernel_config": fp32})
    assert not wa.applies((wh,), {"compute_kernel_config": fp32})  # WH keeps fp32 dest acc
    assert not wa.applies((q,), {"compute_kernel_config": bf16})
    assert not wa.applies((q,), {})
    seen = {}
    wa.rewrite(lambda *a, **k: seen.update(k), (q,), {"compute_kernel_config": fp32, "epsilon": 1e-6})
    c = seen["compute_kernel_config"]
    assert (
        not c.fp32_dest_acc_en
        and c.math_fidelity == ttnn.MathFidelity.HiFi2
        and c.packer_l1_acc
        and seen["epsilon"] == 1e-6
    )


def test_quasar_experimental_multiply_covers_both_aliases(monkeypatch):
    import ttnn

    from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O

    by_target = {w.target: w for w in O.WORKAROUNDS if w.name.startswith("quasar_experimental_mul")}
    assert set(by_target) == {"ttnn.mul", "ttnn.multiply"}
    monkeypatch.setattr(ttnn.experimental.quasar, "multiply", lambda *a, **k: ("q-mul", a, k))
    for wa in by_target.values():
        assert wa.rewrite(None, ("a", "b"), {"dtype": "d"}) == ("q-mul", ("a", "b"), {"dtype": "d"})
