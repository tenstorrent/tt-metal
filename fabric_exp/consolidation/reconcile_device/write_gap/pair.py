"""pair.py: study v4txa11 bh_2d router images against the port's (6dad5fec638) by router class."""
import glob, json, os, re, collections
def named(h): return dict((k, int(v)) for k, v in re.findall(r'"([A-Z0-9_]+)"\s*,\s*(\d+)', open(h).read()))
KEYS = ["MY_DIRECTION", "MY_ETH_CHANNEL", "NUM_DOWNSTREAM_SENDERS_VC0", "DISABLE_RX_CH0_FORWARDING"]
def study():
    out = collections.defaultdict(list)
    for d in glob.glob("/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/v4txa11_O3/bh_2d__*"):
        n = named(os.path.join(d, "named_ct_arg_map_generated.h"))
        for proc in ("active_erisc", "subordinate_active_erisc"):
            e = os.path.join(d, proc, proc + ".elf")
            if os.path.exists(e):
                out[(proc,) + tuple(n.get(k) for k in KEYS)].append((e, n))
    return out
def port():
    out = collections.defaultdict(list)
    hdr = {os.path.basename(os.path.dirname(h)): h for h in glob.glob("/home/aagarwal/.tt-metal-port-gap-cov/cases/bh_2d/cache/*/kernels/fabric_erisc_router/*/named_ct_arg_map_generated.h")}
    for e in glob.glob("/home/aagarwal/.tt-metal-port-gap-cov/builds/gap/bh_2d__kernels__fabric_erisc_router__*/*.elf"):
        m = re.search(r"fabric_erisc_router__(\d+)__(\w+)/", e)
        n = named(hdr[m.group(1)])
        out[(m.group(2),) + tuple(n.get(k) for k in KEYS)].append((e, n))
    return out
s, p = study(), port()
res = []
for k in sorted(set(s) | set(p)):
    res.append({"key": k, "study": [e for e, _ in s.get(k, [])], "port": [e for e, _ in p.get(k, [])]})
    print(k, len(s.get(k, [])), len(p.get(k, [])))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "pairs.json"), "w"), indent=1)
