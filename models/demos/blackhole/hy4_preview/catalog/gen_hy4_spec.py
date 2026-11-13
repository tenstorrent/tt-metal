# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator of the tt-inference-server catalog entries for Hy4-Preview (skeleton; a NEW file, nothing in tt-inference-server is edited).

How the DeepSeek-V4.1-Flash entries are made (tt-inference-server worktree wt/ax_img6_ttis, release 0.24.0 + image 5): there is no gen_*.py. The catalog is
YAML, ``workflows/model_specs/dev/llm.yaml`` (one ``- weights: ... impl: ... device_model_specs: [...]`` block per entry), plus an ``ImplSpec`` per impl name in
``workflows/model_spec.py`` (``deepseek_v41_flash_b8_1m_impl`` ... registered in ``_IMPL_REGISTRY``). ``scripts/build_local_dsv41_image.sh`` exports the merged catalog
with ``export_model_specs_json(MODEL_SPECS, ...)`` into ``model_spec.json`` (image 5: ``release/dsv41_flash_img5/model_spec.json``), baked at
``/home/container_app_user/model_specs/model_spec.json``. This module renders the two Hy4 blocks (B=4 primary and B=8 secondary) in that YAML format, and the two ImplSpec snippets.

Run: ``python -m models.demos.blackhole.hy4_preview.catalog.gen_hy4_spec [--out DIR]`` (writes llm_hy4_preview.yaml and impl_specs_hy4_preview.py.txt, to paste/merge by the release task).

Conventions (copied from the DSV4 entries):
* code default OFF / catalog ON: every opt-in model feature is an env var that the model code reads with default OFF; the catalog entry sets it to "1".
* every field that is not a fact of the checkpoint or of decision D0-x carries ``# UNVERIFIED`` in the rendered YAML (``STATUS`` below is the single list).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HF_ID = "tencent/Hy4-preview"
DEVICE = "BLACKHOLE_GALAXY"
MAX_CTX = 1048576  # max_position_embeddings (checkpoint fact). Whether B=16 x 1M fits is the open D-CAP decision (port plan 2c): pass max_context= to override.

V = "VERIFIED"
U = "UNVERIFIED"


@dataclass(frozen=True)
class Variant:
    impl_id: str
    impl_name: str
    users: int  # max_concurrency = batch B
    up: int  # prefill users per mesh row (Up = U = B / 4)
    max_partial: int  # DSV41_VLLM_MAX_PARTIAL analogue
    spec_k: int = 3  # README MTP recipe


# D3-1: PRIMARY B=4 x 1M, SECONDARY B=8 (bfp4); B=16 dropped.
B4 = Variant("hy4_preview_b4", "hy4-preview-b4", 4, 1, 4)
B8 = Variant("hy4_preview_b8", "hy4-preview-b8", 8, 2, 8)
VARIANTS = (B4, B8)
# Per-variant max_context. B=4 is the primary contract (1M, KV SCALED_FP8 per D3-1: still an assumption, kv-design not final). B=8: no kv-design number guaranteed
# yet (checked: $PROGRAM/kv-design empty at 2026-11-13), so 1M marked UNVERIFIED pending bfp4 KV; replace by the largest guaranteed value when kv-design lands.
MAX_CTX_BY_IMPL = {"hy4-preview-b4": MAX_CTX, "hy4-preview-b8": MAX_CTX}
CTX_STATUS = {"hy4-preview-b4": V, "hy4-preview-b8": U}


# (env name, catalog value, status, comment). Names are HY4_* (the Hy4 model code reads them; code default OFF unless noted). Values and even the names are
# placeholders until the model tracks land: the tracks own their flags, this list is where the release task collects them.
def env_vars(v: Variant) -> List[Tuple[str, str, str, str]]:
    return [
        ("HF_MODEL", HF_ID, V, "hf id"),
        ("MESH_DEVICE", "(4, 8)", V, "Galaxy 4x8 (D0-2)"),
        ("HY4_LAYERS", "0-77", U, "layer range; the full model (analogue of DSV41_LAYERS)"),
        (
            "HY4_CKPT",
            "/ckpt",
            U,
            "container path of the checkpoint (ro mount); read by hy4_preview.ckpt_stamp / weights",
        ),
        (
            "HY4_WEIGHT_CACHE",
            "/cache/weights",
            U,
            "NVMe cache dir, stamped with the checkpoint id (ckpt_stamp.ensure_stamp)",
        ),
        ("TT_METAL_CACHE", "/cache/kernels", U, "JIT kernel cache"),
        (
            "EXTRA_MODELS_DIR",
            "../../tt-metal/models/demos/blackhole",
            V,
            "plugin registers the bundle folder hy4_preview (vllm_metadata.json, to be added by the plugin track)",
        ),
        ("HY4_VLLM_CHUNK", "512", U, "prefill chunk = APC grid (port plan: hit grid stays 512)"),
        ("HY4_PREFILL_UP", str(v.up), U, "Up = U (D0-2): users per mesh row of the prefill trace"),
        ("HY4_VLLM_MAX_PARTIAL", str(v.max_partial), U, "capabilities max_partial_prefills"),
        ("HY4_APC", "1", U, "model-owned automatic prefix caching, code default OFF, catalog ON"),
        (
            "HY4_APC_PAGED",
            "1",
            U,
            "device-resident block-paged APC, code default OFF, catalog ON (geometry owned by the kv track: page 128 x 656 B?)",
        ),
        ("HY4_SPEC", "1", U, "native MTP speculative decode, code default OFF, catalog ON"),
        ("HY4_SPEC_K", str(v.spec_k), U, "num_speculative_tokens (README recipe k=3)"),
        ("HY4_EXPERT_DTYPE", "bfp4", V, "routed experts bfp4: mandatory (D0-3)"),
        (
            "HY4_KV_FORMAT",
            "scaled_fp8",
            U,
            "KV pool format; open decision D-CAP (bf16 / scaled_fp8 / fp8 / bfp4), see port plan section 2",
        ),
        ("HY4_TRACE_REGION", "1200000000", U, "copied from the DSV4 entries"),
    ]


def vllm_args(v: Variant, max_ctx: int) -> List[Tuple[str, str, str, str]]:
    return [
        ("block_size", '"128"', U, "page size; DSV4 value, kv track decides"),
        (
            "hf_config_path",
            "../../tt-metal/models/demos/blackhole/hy4_preview/vllm_config",
            U,
            "serving shim config.json (llama-shaped, like the DSV4 one); to be created by the plugin track",
        ),
        ("tokenizer", "/ckpt", V, "tokenizer.json is in the checkpoint"),
        ("max_num_seqs", f'"{v.users}"', V, "B (D0-2)"),
        ("max_model_len", f'"{max_ctx}"', U, "1M per user is the contract (D0-2); fit under D-CAP not shown"),
        ("max_num_batched_tokens", f'"{v.max_partial * 4096}"', U, "DSV4 rule: partial prompts x 4096"),
        ("long_prefill_token_threshold", "4096", U, "DSV4 value"),
        ("enable-prefix-caching", "true", U, "APC on in the catalog (code default off)"),
        ("enable-auto-tool-choice", "true", V, "tool calling is in scope (D0-6)"),
        (
            "tool-call-parser",
            "hy_v4",
            U,
            "name from the HF model card / vllm registry; absent from our image until checked (fallback: register in plugin entrypoints._TT_TOOL_PARSERS)",
        ),
        ("reasoning-parser", "hy_v4", U, "same as above (_TT_REASONING_PARSERS)"),
        (
            "default-chat-template-kwargs",
            '\'{"reasoning_effort": "high"}\'',
            V,
            "chat_template reasoning_effort in {high, no_think}, default high; per request: chat_template_kwargs",
        ),
        ("hf-overrides", '\'{"architectures":["TTHYV4ForCausalLM"]}\'', U, "plugin prefixes TT to HYV4ForCausalLM"),
        (
            "speculative_config",
            '\'{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": %d}\''
            % v.spec_k,
            U,
            "DSV4 mechanism; MTP drafter owned by the model",
        ),
    ]


OVERRIDE_TT_CONFIG = [
    ("fabric_config", "FABRIC_1D_RING", U, "copied from DSV4"),
    ("l1_small_size", "16384", U, "copied from DSV4"),
    ("trace_region_size", "1200000000", U, "copied from DSV4"),
    ("sample_on_device_mode", "all", U, "copied from DSV4"),
]

METADATA = {"tool_call_parser_name": ("hy_v4", U), "reasoning_parser_name": ("hy_v4", U)}


# Fields of the entry itself: (key, value, status, comment)
def _impl_of(users: int) -> str:
    return f"hy4-preview-b{users}"


def entry_fields(v: Variant, max_ctx: int) -> List[Tuple[str, str, str, str]]:
    return [
        ("max_concurrency", str(v.users), V, "B = 4 mesh rows x Up users"),
        (
            "max_context",
            str(max_ctx),
            CTX_STATUS.get(v.impl_name, U),
            "per request (prompt + output); native 1M; B=8 pending bfp4 KV (D3-1)",
        ),
        ("max_tokens_all_users_override", str(v.users * max_ctx), U, "benchmark sizing only, B x max_context"),
        ("default_impl", "false", V, "no default: Hy4 is not the default model of the device"),
        (
            "tensor_cache_timeout",
            "7200.0",
            U,
            "first boot builds a bf16 -> bfp4 cache for 1.5 TB; DSV4 needed 5444 s (fp8)",
        ),
    ]


def _line(ind: int, key: str, val: str, status: str, comment: str) -> str:
    tag = "" if status == V else "UNVERIFIED: "
    c = f"  # {tag}{comment}" if (tag or comment) else ""
    return " " * ind + f"{key}: {val}{c}"


def render_entry(v: Variant, max_ctx: Optional[int] = None) -> str:
    max_ctx = max_ctx or MAX_CTX_BY_IMPL[v.impl_name]
    out: List[str] = []
    out.append(
        f"# Hy4-Preview, {v.users} users x {max_ctx} tokens ({'PRIMARY' if v.users == 4 else 'SECONDARY'}, D3-1) (impl {v.impl_name}). GENERATED by hy4_preview/catalog/gen_hy4_spec.py: edit the generator, not this text."
    )
    out.append("- weights:")
    out.append(f"    - {HF_ID}")
    out.append(f"  impl: {v.impl_id}")
    out.append("  inference_engine: VLLM")
    out.append("  model_type: LLM")
    out.append("  supported_modalities:")
    out.append("    - text  # checkpoint is text-only (D0-6)")
    out.append("  device_model_specs:")
    out.append(f"    - device: {DEVICE}")
    for k, val, st, c in entry_fields(v, max_ctx):
        out.append(_line(6, k, val, st, c))
    out.append("      env_vars:")
    for k, val, st, c in env_vars(v):
        q = f'"{val}"' if not val.startswith(("/", "../")) and k != "HF_MODEL" else val
        out.append(_line(8, k, q, st, c))
    out.append("      vllm_args:")
    for k, val, st, c in vllm_args(v, max_ctx):
        out.append(_line(8, k, val, st, c))
    out.append("      override_tt_config:")
    for k, val, st, c in OVERRIDE_TT_CONFIG:
        out.append(_line(8, k, val, st, c))
    out.append("  status: EXPERIMENTAL")
    out.append("  has_builtin_warmup: true  # UNVERIFIED: depends on the model warmup methods")
    out.append("  metadata:")
    out.append(f"    {HF_ID}:")
    for k, (val, st) in METADATA.items():
        out.append(_line(6, k, val, st, "parser name in the vLLM registry (check in the image)"))
    return "\n".join(out) + "\n"


def render_impl_specs() -> str:
    out = [
        "# Paste into tt-inference-server workflows/model_spec.py next to the deepseek_v41_flash_* ImplSpecs, and register in _IMPL_REGISTRY.\n"
    ]
    for v in VARIANTS:
        out.append(
            f"{v.impl_id}_impl = ImplSpec(\n"
            f'    impl_id="{v.impl_id}",\n'
            f'    impl_name="{v.impl_name}",\n'
            '    repo_url="https://github.com/tenstorrent/tt-metal",\n'
            '    code_path="models/demos/blackhole/hy4_preview",\n'
            ")\n"
        )
    out.append("# _IMPL_REGISTRY additions:")
    for v in VARIANTS:
        out.append(f'#     "{v.impl_id}": {v.impl_id}_impl,')
    return "\n".join(out) + "\n"


def unverified_fields(max_ctx: int = MAX_CTX) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for v in VARIANTS:
        names = []
        for grp in (entry_fields(v, max_ctx), env_vars(v), vllm_args(v, max_ctx), OVERRIDE_TT_CONFIG):
            names += [k for k, _, st, _ in grp if st == U]
        names += [k for k, (_, st) in METADATA.items() if st == U]
        res[v.impl_name] = names
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=None,
        help="directory for llm_hy4_preview.yaml and impl_specs_hy4_preview.py.txt (default: print)",
    )
    ap.add_argument("--max-context", type=int, default=None, help="override for all variants")
    a = ap.parse_args(argv)
    mc = a.max_context
    yaml_text = "\n".join(render_entry(v, mc) for v in VARIANTS)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "llm_hy4_preview.yaml"), "w") as f:
            f.write(yaml_text)
        with open(os.path.join(a.out, "impl_specs_hy4_preview.py.txt"), "w") as f:
            f.write(render_impl_specs())
    else:
        print(yaml_text)
        print(render_impl_specs())
    for name, fields in unverified_fields(mc or MAX_CTX).items():
        print(f"{name}: {len(fields)} UNVERIFIED fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
