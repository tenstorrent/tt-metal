"""Print the collected node ids that match any pattern line (substrings joined by ' && '). usage: pick.py <ids> <patterns>"""
import sys
ids = [l.strip() for l in open(sys.argv[1]) if "::" in l]
for pat in (l.strip() for l in open(sys.argv[2])):
    if not pat or pat.startswith("#"):
        continue
    parts = pat.split(" && ")
    hit = [i for i in ids if all(p in i for p in parts)]
    print(f"# {len(hit)} for {pat}", file=sys.stderr)
    for i in hit[:4]:
        print(i)
