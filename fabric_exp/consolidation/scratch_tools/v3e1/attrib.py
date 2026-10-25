import sys, re, collections
lines = open(sys.argv[1]).read().split("\n")
recs = []; cur = None
i = 0
while i < len(lines):
    l = lines[i]
    if re.match(r"^0x[0-9a-f]+$", l):
        cur = [int(l, 16), []]; recs.append(cur); i += 1; continue
    if cur is not None and l and i + 1 < len(lines):
        cur[1].append((l, lines[i+1])); i += 2; continue
    i += 1
# outermost frames are last; attribute each instruction to the chain of function names (innermost first)
keys = ["receiver_channel_step_impl", "run_receiver_channel_step", "execute_chip_unicast_to_local_chip_impl", "forward_2d_dispatch", "admit_2d_dispatch",
        "run_fabric_edm_main_loop", "kernel_main", "teardown", "execute_main_loop", "flush_completion_credits", "accumulate_completion_credit"]
cnt = collections.Counter()
for a, fr in recs:
    names = [f for f, loc in fr]
    tag = None
    chain = " < ".join(n.split("(")[0].split("<")[0].strip() for n in names)
    # pick: which receiver step instance
    stepl = [ (f,loc) for f,loc in fr if "run_receiver_channel_step_impl" in f]
    if stepl:
        m = re.search(r"run_receiver_channel_step_impl<\(unsigned char\)(\d)", stepl[0][0])
        ch = m.group(1) if m else "?"
        inner = names[0].split("(")[0]
        sub = "local" if any("execute_chip_unicast" in n for n in names) else ("fwd" if any("forward_" in n or "send_payload" in n for n in names) else ("admit" if any("admit_2d" in n for n in names) else "rest"))
        cnt[f"rx{ch}:{sub}"] += 4
    elif any("run_fabric_edm_main_loop" in n for n in names):
        cnt["mainloop-other"] += 4
    elif any("kernel_main" in n for n in names):
        cnt["kernel_main-init"] += 4
    else:
        cnt[names[-1].split("(")[0][:60] if names else "?"] += 4
for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"{v:6d} {k}")
print(sum(cnt.values()))
