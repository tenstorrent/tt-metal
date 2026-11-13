# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the Hy4 catalog generator: the rendered YAML parses, has the DSV4-entry structure, and marks unverified fields."""

import yaml

from models.demos.blackhole.hy4_preview.catalog import gen_hy4_spec as G


def _parse():
    text = "\n".join(G.render_entry(v) for v in G.VARIANTS)
    return text, yaml.safe_load(text)


def test_structure():
    _, blocks = _parse()
    assert [b["impl"] for b in blocks] == ["hy4_preview_b8", "hy4_preview_b16"]
    for b, users in zip(blocks, (8, 16)):
        assert b["weights"] == ["tencent/Hy4-preview"]
        assert b["supported_modalities"] == ["text"]
        d = b["device_model_specs"][0]
        assert d["device"] == "BLACKHOLE_GALAXY" and d["max_concurrency"] == users and d["max_context"] == 1048576
        assert d["env_vars"]["HY4_EXPERT_DTYPE"] == "bfp4" and d["env_vars"]["HF_MODEL"] == "tencent/Hy4-preview"
        assert int(d["env_vars"]["HY4_PREFILL_UP"]) == users // 4  # Up = U
        assert d["vllm_args"]["tool-call-parser"] == "hy_v4" and d["vllm_args"]["reasoning-parser"] == "hy_v4"
        assert d["vllm_args"]["max_num_seqs"] == str(users)
        assert "reasoning_effort" in d["vllm_args"]["default-chat-template-kwargs"]
        assert b["metadata"]["tencent/Hy4-preview"] == {
            "tool_call_parser_name": "hy_v4",
            "reasoning_parser_name": "hy_v4",
        }


def test_code_default_off_flags_are_on_in_catalog():
    _, blocks = _parse()
    env = blocks[0]["device_model_specs"][0]["env_vars"]
    for k in ("HY4_APC", "HY4_APC_PAGED", "HY4_SPEC"):
        assert env[k] == "1"


def test_every_unverified_field_is_marked():
    text, _ = _parse()
    for name, fields in G.unverified_fields().items():
        assert fields  # nothing may be silently verified here
    for line in text.splitlines():
        for k in ("tool-call-parser", "HY4_KV_FORMAT", "max_model_len", "block_size"):
            if line.strip().startswith(k + ":"):
                assert "UNVERIFIED" in line, line


def test_impl_specs_text():
    t = G.render_impl_specs()
    assert 'impl_name="hy4-preview-b8"' in t and 'impl_name="hy4-preview-b16"' in t
    compile(t.split("# _IMPL_REGISTRY")[0].replace("ImplSpec(", "dict(").split("\n", 1)[1], "<impl>", "exec")


def test_main_writes(tmp_path):
    assert G.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "llm_hy4_preview.yaml").stat().st_size > 1000
