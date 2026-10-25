import re, sys, collections
txt = open(sys.argv[1]).read()
sec = None
counts = collections.Counter(); per = collections.defaultdict(collections.Counter)
rows = 0
for line in txt.splitlines():
    m = re.match(r'## (\d)\. ', line)
    if m: sec = int(m.group(1)); continue
    if sec is None or sec > 6: continue
    if not line.startswith('| ') or line.startswith('| ID') or line.startswith('| ---'): continue
    # split on unescaped pipes
    cells = [c.strip() for c in re.split(r'(?<!\\)\|', line)[1:-1]]
    if len(cells) != 7:
        print('BAD', sec, len(cells), line[:80]); continue
    st = re.search(r'\*\*([A-Z-]+)\*\*', cells[5])
    if not st: print('NOSTATUS', line[:80]); continue
    counts[st.group(1)] += 1; per[sec][st.group(1)] += 1; rows += 1
print(rows, dict(counts))
for s in sorted(per): print(s, dict(per[s]))
