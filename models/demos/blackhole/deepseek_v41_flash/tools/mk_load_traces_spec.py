"""mk_load_traces_spec.py <catalog.json> <out.json> [K=V env overrides ...]

Adds the catalog entries `deepseek_v41_flash_b16_1m_lt` and `deepseek_v41_flash_b8_1m_lt` (impls `deepseek-v41-flash-b16-1m-lt`, `deepseek-v41-flash-b8-1m-lt`) to a combined catalog: copies of the
B=16 / B=8 1M entries with the load-based decode traces switched on (DSV41_LOAD_TRACES=1: next to the full trace the server captures adaptive runner sets for the first Ub < U users of every mesh
row, DSV41_LOAD_TRACE_BUCKETS = 1,2 at B=16 (= B=4 and B=8 traces) and 1 at B=8 (= B=4)), the trace region bumped for the extra traces and the small-T fuse-tail / attention-front switches of the B=4
trace (DSV41_FUSE_TAIL_SMALL, DSV41_ATTN_FRONT_SMALL). The shipped entries stay untouched (the new impls are not default). See dsv4-logs/unified_stack/LOAD_TRACES_PLAN.md.
"""
import copy
import json
import sys

SPECS = {
    # (source entry, new entry, buckets, trace region bytes)
    "b16": (
        "deepseek_v41_flash_b16_1m",
        "deepseek_v41_flash_b16_1m_lt",
        "1",
        "1000000000",
    ),  # (measured: ~14 MiB/bank of DRAM per extra runner (k) set; two buckets do not fit without bfp4)
    "b8": ("deepseek_v41_flash_b8_1m", "deepseek_v41_flash_b8_1m_lt", "1", "1200000000"),
}


def add(full, extra=None):
    v = full["model_specs"]["deepseek-ai/DeepSeek-V4.1-Flash"]["BLACKHOLE_GALAXY"]["vLLM"]
    for key, (src, new, buckets, region) in SPECS.items():
        e = copy.deepcopy(v[src])
        e["model_id"] = e["model_id"].replace("-1m_", "-1m-lt_") if "-1m_" in e["model_id"] else e["model_id"] + "-lt"
        e["impl"]["impl_id"] = new
        e["impl"]["impl_name"] = e["impl"]["impl_name"] + "-lt"
        d = e["device_model_spec"]
        d["default_impl"] = False
        env_add = {
            "DSV41_LOAD_TRACES": "1",
            "DSV41_LOAD_TRACE_BUCKETS": buckets,
            "DSV41_TRACE_REGION": region,
            "DSV41_FUSE_TAIL_SMALL": "1",
            "DSV41_ATTN_FRONT_SMALL": "1",
            "DSV41_DROP_LAST_ROUTING": "1",
        }
        env_add.update(extra or {})
        e["env_vars"].update(env_add)
        d.setdefault("env_vars", e["env_vars"])
        d["env_vars"].update(env_add)
        tt = d.get("override_tt_config")
        if isinstance(tt, dict) and "trace_region_size" in tt:
            tt["trace_region_size"] = int(region)
        va = d["vllm_args"].get("additional_config")
        if isinstance(va, str) and "trace_region_size" in va:
            cfg = json.loads(va)
            cfg["tt"]["trace_region_size"] = int(region)
            d["vllm_args"]["additional_config"] = json.dumps(cfg)
        v[new] = e
    return full


def main():
    src, out = sys.argv[1], sys.argv[2]
    extra = dict(a.split("=", 1) for a in sys.argv[3:])
    json.dump(add(json.load(open(src)), extra), open(out, "w"), indent=1)
    print(out, [s[1] for s in SPECS.values()], sorted(extra))


if __name__ == "__main__":
    main()
