import re,sys
def load(v):
    ins=[]
    for l in open(f"{v}.dis"):
        if not re.match(r'^\s+[0-9a-f]+:\t',l): continue
        p=l.rstrip("\n").split("\t")
        a=int(p[0].strip().rstrip(":"),16); op=p[2].strip() if len(p)>2 else ""; arg=p[3].strip() if len(p)>3 else ""
        ins.append((a,op,arg))
    return ins
def target(arg):
    m=re.search(r'\b([0-9a-f]{4,8}) <',arg)
    return int(m.group(1),16) if m else None
if __name__=="__main__":
    for v in sys.argv[1:]:
        ins=load(v)
        km=[x for x in ins if 0xc910<=x[0]<0xe100]
        print(v)
        for a,op,arg in km:
            t=target(arg)
            if t is not None and t<=a and t>=0xc910 and (op.startswith("b") or op=="j"):
                print(f"   back-edge {a:x} -> {t:x}  {op:6s} span {(a-t)//4}")
