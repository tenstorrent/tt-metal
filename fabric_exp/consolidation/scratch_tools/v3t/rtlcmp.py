import re, sys, glob, difflib, os
S="/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t"
A=f"{S}/t3r/10856749301170935304/subordinate_active_erisc"; B=f"{S}/tmr/10856749301170935304/subordinate_active_erisc"
fn = sys.argv[1] if len(sys.argv) > 1 else "_start"
def sect(path):
    s=open(path, errors="replace").read()
    i=s.find(f";; Function {fn} ")
    if i<0: return None
    j=s.find("\n;; Function", i+10)
    t=s[i:j if j>0 else None]
    t=t.split("\n",1)[1]
    # normalize insn uids, label numbers, decl uids, block numbers
    t=re.sub(r"D\.\d+","D",t); t=re.sub(r"\b(insn|jump_insn|call_insn|code_label|note|barrier|debug_insn|jump_table_data)(/\w)? \d+ \d+ \d+","\\1 U U U",t)
    t=re.sub(r"\(code_label \d+\)","(code_label L)",t); t=re.sub(r"L\d+","L",t); t=re.sub(r"\[\d+ [^\]]*\]","[M]",t)
    t=re.sub(r"\(nil\)","",t)
    return t.split("\n")
for f in sorted(glob.glob(A+"/*ltrans0.ltrans.*r.*")):
    suf=f.split("ltrans0.ltrans.")[1]
    g=B+"/subordinate_active_erisc.elf.ltrans0.ltrans."+suf
    if not os.path.exists(g): continue
    a,b=sect(f),sect(g)
    if a is None or b is None: continue
    d=[l for l in difflib.unified_diff(a,b,lineterm="",n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
    print(f"{suf:22s} {len(a):6d} {len(b):6d} diff {len(d)}")
    if d and "--first" in sys.argv:
        print("\n".join(d[:30])); break
