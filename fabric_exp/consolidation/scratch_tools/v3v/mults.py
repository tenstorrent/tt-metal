import sys, re
for path in sys.argv[1:]:
    s = open(path).read()
    out = []
    for m in re.finditer(r";; Function (\S+) \(([^,]+),", s):
        start = m.end(); nxt = s.find("\n;; Function", start); body = s[start:nxt if nxt > 0 else None]
        if "run_main_loop" not in m.group(2):
            continue
        mults = re.findall(r"= \S+ \* (\d+);", body)
        bad = [x for x in mults if int(x) & (int(x) - 1)]
        rr = len(re.findall(r"= \S+ \* [a-z_]\S*;", body))
        out.append("%s: non-pow2 const mults %s; var*var %d" % (m.group(1), bad, rr))
    print(path.split("/")[-4], out)
