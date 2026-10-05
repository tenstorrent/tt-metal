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
    }


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
