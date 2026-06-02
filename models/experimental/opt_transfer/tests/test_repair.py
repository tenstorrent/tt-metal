from models.experimental.opt_transfer.repair import build_diagnosis, localize_culprit


def test_localize_identifies_the_breaking_fusion():
    applied = ["qkv_merge", "ffn_fuse", "norm_fuse"]

    def pcc_with(disabled: set) -> float:
        return 0.999 if "ffn_fuse" in disabled else 0.80

    culprit = localize_culprit(applied, pcc_with, threshold=0.99)
    assert culprit == "ffn_fuse"


def test_localize_returns_none_when_all_pass():
    def pcc_with(disabled):
        return 0.999

    assert localize_culprit(["a", "b"], pcc_with, threshold=0.99) is None


def test_diagnosis_per_block_axis():
    d = build_diagnosis(
        node="ffn_fuse",
        per_block_pcc=0.80,
        tf_pcc=None,
        free_run_divergence_frac=None,
        config_tried={"dtype": "bf16"},
    )
    assert d.axis == "per_block_pcc" and d.measured == 0.80


def test_diagnosis_accumulation_axis_when_tf_ok_but_freerun_diverges():
    d = build_diagnosis(
        node="qkv_merge",
        per_block_pcc=0.999,
        tf_pcc=0.999,
        free_run_divergence_frac=0.2,
        config_tried={},
    )
    assert d.axis == "long_decode_drift"
