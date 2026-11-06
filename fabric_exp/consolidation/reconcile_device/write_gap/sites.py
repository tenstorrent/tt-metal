"""sites.py: per-packet path pieces (by inline chain) of a router's run(): instructions, L1 loads, stack loads, stores,
NoC command-buffer register writes (NOC_CMD_BUF_WRITE_REG), and the piece's first address (placement)."""
import collections, json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from annot import annotate
E0 = [("send<0u>", "worker send (send<0u>)"), ("send<1u>", "forwarding send ch1"), ("send<2u>", "forwarding send ch2"),
      ("send<3u>", "forwarding send ch3"), ("credit_worker", "worker credit"), ("preset", "worker credit"),
      ("visit_worker", "worker visit (other)"), ("visit_channel_with_work", "forwarding visit (other)"),
      ("channels_with_packets", "pass-start reads"), ("check_worker_connection", "connection check")]
E1 = [("local_write<true>", "local write (preset)"), ("local_write<false>", "local write (whole)"), ("local_atomic_inc", "atomic"),
      ("forward", "forward"), ("decode_action", "route decode"), ("admit", "admission"),
      ("accumulate_completion_credit", "credit accumulate"), ("credit_completions", "completion proof / due"),
      ("send_credit_frame", "credit frame"), ("retire", "retire"), ("deliver", "deliver (dispatch)"),
      ("receive", "receive (header)")]
def site(chain, table):
    for f, _ in chain:
        for pat, name in table:
            if f.startswith(pat) or (pat in ("forward", "preset") and pat in f):
                return name
    return None
def count(elf, proc):
    fn, table = (r"sender::run[<(]", E0) if proc == "active_erisc" else (r"receiver::run[<(]", E1)
    _, rows = annotate(elf, fn)
    c = collections.defaultdict(collections.Counter); first = {}
    for a, t, ch in rows:
        s = site(ch, table)
        if not s: continue
        op = t.split()[0]; c[s]["insns"] += 1
        if op in ("lw", "lh", "lhu", "lb", "lbu"): c[s]["sp load" if "(sp)" in t else "load"] += 1
        if op in ("sw", "sh", "sb"): c[s]["sp store" if "(sp)" in t else "store"] += 1
        if any(f.startswith("NOC_CMD_BUF_WRITE_REG") for f, _ in ch): c[s]["cmd reg writes"] += 1
        first.setdefault(s, a)
    return c, first, len(rows)
pairs = json.load(open(sys.argv[1]))
for want in sys.argv[2:]:
    p = [p for p in pairs if str(p["key"]) == want][0]
    (cs, fs, ns), (cp, fp, np_) = count(p["study"][0], p["key"][0]), count(p["port"][0], p["key"][0])
    print(f"== {p['key']}: run() study {ns} / port {np_} insns")
    for s in sorted(set(cs) | set(cp)):
        f = lambda x: f"{x['insns']:3d} i, {x['load']} ld, {x['sp load']} sp-ld, {x['store']} st, {x['sp store']} sp-st, {x['cmd reg writes']} cmd"
        print(f"   {s:28s} study {f(cs[s])}  | port {f(cp[s])}" + ("   <-- differs" if cs[s] != cp[s] else ""))
