"""mk_b16_1m_s5_spec.py <catalog.json> <out.json> [K=V env overrides ...]

Adds the catalog entry `deepseek_v41_flash_b16_1m` (16 users x 1,048,576 tokens, adaptive spec k=3/5, Engram on) to a stack5 model_spec JSON
(e.g. dsv4-logs/rc5/model_spec_rc5c_stack5.json). It is derived from the B=8 / 1M entry `deepseek_v41_flash_b8_1m` and is NOT the default impl:
select it with `--impl deepseek-v41-flash-b16-1m`. Default off for everything else: the B=8 entry is left untouched.

Why the overrides (see dsv4-logs/b16_1m_plan/PLAN.md): at 1M the fp8 KV pool costs 160 MiB/bank per user, so B=16 needs 320 MiB/bank more than B=8. The stack5 B=8
build has only 187 MiB/bank free, so everything that is dead weight at B=16 is switched off:
  * DSV41_APC_PAGED / _XROW / DSV41_APC_IMAGES: paged device APC is unsupported when Up < U (B=16: Up=2, U=4); the host-tier APC stays on (65 MiB/bank).
  * DSV41_ATTN_FRONT*, DSV41_SKINNY_*: duplicate blaze/bank-sharded weight copies of the T in (8, 12) / T <= 8 / T <= 16 paths; at B=16 the verify rows are
    T = 4 x (1 + k) = 16 or 24 (about 143 MiB/bank).
  * DSV41_FUSE_TAIL: T in (8, 12) only.
  * DSV41_MHC_PRE24 stays 0 (the B=16 k=5 decode gate fails with it).
"""
import copy
import json
import sys

SRC_IMPL = "deepseek_v41_flash_b8_1m"
NEW_IMPL = "deepseek_v41_flash_b16_1m"
MAX_LEN = 1048576
BATCH = 16

# env applied to the entry (both entry.env_vars and device_model_spec.env_vars)
B16_ENV = {
    "MAX_NUM_SEQS": str(BATCH),
    "MAX_MODEL_LENGTH": str(MAX_LEN),
    "DSV41_VLLM_STATS_EVERY": "16",
    "DSV41_SPEC_MAX_CTX": str(MAX_LEN + 256),
    "DSV41_MHC_PRE24": "0",
}
# dead weight / unsupported at B=16 (value "0" so that a later `setdefault` cannot turn them back on)
B16_OFF = (
    "DSV41_APC_PAGED",
    "DSV41_APC_PAGED_XROW",
    "DSV41_APC_IMAGES",
    "DSV41_ATTN_FRONT",
    "DSV41_ATTN_FRONT_ARM",
    "DSV41_ATTN_FRONT_XSPLIT",
    "DSV41_SKINNY_ARM",
    "DSV41_SKINNY_WOAB",
    "DSV41_SKINNY_SHARED",
    "DSV41_FUSE_TAIL",
)
# knobs that only matter when the switch they refine is on: dropped from the B=16 env
B16_DROP = (
    "DSV41_ATTN_FRONT_RING",
    "DSV41_SKINNY_CT",
    "DSV41_SKINNY_SBK1",
    "DSV41_SKINNY_SBK2",
    "DSV41_SKINNY_XL1",
    "DSV41_APC_GRID",
)


# ALL-ON variant (--all-on, impl deepseek_v41_flash_b16_1m_allon): every stack5 optimization stays ON as in the B=8 entry; DRAM is freed by the default-off levers of
# tt/dram_levers.py instead (dsv4-logs/unified_stack/DRAM_LEDGER.md): copy guard (no attention-front / skinny shared copy when no verify row count T = 4 x (1 + k) can use it),
# embedding table sharded over the columns, trace region 1.0e9; only the paged device APC (unsupported at Up < U) is off.
NEW_IMPL_ALLON = "deepseek_v41_flash_b16_1m_allon"
ALLON_ENV = {"DSV41_COPY_GUARD": "1", "DSV41_EMBED_SHARD": "1", "DSV41_TRACE_REGION": "1000000000"}
ALLON_OFF = ("DSV41_APC_PAGED", "DSV41_APC_PAGED_XROW", "DSV41_APC_IMAGES")


def b16_env(base_env, extra=None, all_on=False):
    env = dict(base_env)
    if all_on:
        env.update({k: "0" for k in ALLON_OFF})
        env.update(B16_ENV)
        env.update(ALLON_ENV)
        env.update(extra or {})
        return env
    for k in B16_DROP:
        env.pop(k, None)
    env.update({k: "0" for k in B16_OFF})
    env.update(B16_ENV)
    env.update(extra or {})
    return env


def add_b16_entry(full, extra=None, all_on=False):
    """full: catalog dict (schema_version / release_version / model_specs). Returns the same dict with the B=16 entry added (not default)."""
    v = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"]
    e = copy.deepcopy(v[SRC_IMPL])
    new = NEW_IMPL_ALLON if all_on else NEW_IMPL
    e["model_id"] = e["model_id"].replace("b8-1m", "b16-1m")
    e["impl"]["impl_id"] = new
    e["impl"]["impl_name"] = new.replace("_", "-")
    d = e["device_model_spec"]
    d["default_impl"] = False
    d["max_concurrency"] = BATCH
    d["max_context"] = MAX_LEN
    d["max_tokens_all_users_override"] = BATCH * MAX_LEN
    d["vllm_args"]["max_num_seqs"] = str(BATCH)
    d["vllm_args"]["max_model_len"] = str(MAX_LEN)
    env = b16_env(e["env_vars"], extra, all_on)
    e["env_vars"] = env
    d["env_vars"] = b16_env(d.get("env_vars", e["env_vars"]), extra, all_on)
    if (
        all_on
    ):  # the device is opened with the vLLM additional_config / override_tt_config trace_region_size (DSV41_TRACE_REGION is only read by the tools)
        tr = int(ALLON_ENV["DSV41_TRACE_REGION"])
        ac = json.loads(d["vllm_args"]["additional_config"])
        ac["tt"]["trace_region_size"] = tr
        d["vllm_args"]["additional_config"] = json.dumps(ac)
        if isinstance(d.get("override_tt_config"), dict):
            d["override_tt_config"]["trace_region_size"] = tr
    v[new] = e
    return full


def main():
    args = [a for a in sys.argv[1:] if a != "--all-on"]
    all_on = "--all-on" in sys.argv
    src, out = args[0], args[1]
    extra = dict(a.split("=", 1) for a in args[2:])
    full = add_b16_entry(json.load(open(src)), extra, all_on)
    json.dump(full, open(out, "w"), indent=1)
    print(out, NEW_IMPL_ALLON if all_on else NEW_IMPL, sorted(extra))


if __name__ == "__main__":
    main()
