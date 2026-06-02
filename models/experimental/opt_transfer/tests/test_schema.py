from models.experimental.opt_transfer.schema import (
    PatternKind,
    KBEntry,
    FusionProposal,
    Diagnosis,
)


def test_kbentry_roundtrips_to_and_from_dict():
    e = KBEntry(
        id="qkv_merge",
        fused_op="ttnn.experimental.nlp_create_qkv_heads",
        category="attention.qkv",
        pattern_kind=PatternKind.HORIZONTAL_MERGE,
        torch_pattern=["linear", "linear", "linear"],
        signature={"input_rank": 4, "qkv_order": ["q", "k", "v"]},
        config_template={"num_heads": "{H}", "num_kv_heads": "{H}", "transpose_k_heads": False},
        weight_transform="concat_qkv",
        source="models/tt_transformers/tt/attention.py",
        usage_examples=["nlp_create_qkv_heads(qkv, num_heads=32, num_kv_heads=8)"],
        applicability_notes="4D input required; concat order q|k|v",
        status="in_use",
        accumulation_sensitive=False,
    )
    assert KBEntry.from_dict(e.to_dict()) == e


def test_fusionproposal_resolves_config_placeholders():
    p = FusionProposal(
        entry_id="qkv_merge",
        fused_op="ttnn.experimental.nlp_create_qkv_heads",
        matched_nodes=["q_proj", "k_proj", "v_proj"],
        config={"num_heads": "{H}", "num_kv_heads": "{H}", "transpose_k_heads": False},
        weight_transform="concat_qkv",
        rationale="three sibling projections sharing input",
        source="models/tt_transformers/tt/attention.py",
    )
    resolved = p.resolve({"H": 16})
    assert resolved.config == {"num_heads": 16, "num_kv_heads": 16, "transpose_k_heads": False}


def test_diagnosis_axis_is_validated():
    d = Diagnosis(node="q_proj", axis="per_block_pcc", measured=0.97, config_tried={"dtype": "bf16"})
    assert d.axis in ("per_block_pcc", "long_decode_drift")
