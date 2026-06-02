from models.experimental.opt_transfer.repair import localize_culprit


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
