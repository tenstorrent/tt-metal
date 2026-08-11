import json, os, struct, urllib.request, numpy as np
REPO="google/gemma-4-26B-A4B-it"; TOK=os.environ["HF_TOKEN"]
BASE=f"https://huggingface.co/{REPO}/resolve/main/"
def get(url, rng=None):
    h={"Authorization":f"Bearer {TOK}"}
    if rng: h["Range"]=f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300).read()

wm=json.loads(get(BASE+"model.safetensors.index.json"))["weight_map"]
WANT=["model.language_model.layers.0.router.proj.weight",
      "model.language_model.layers.0.router.scale",
      "model.language_model.layers.0.router.per_expert_scale",
      "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
      "model.language_model.layers.0.post_feedforward_layernorm_2.weight"]
byshard={}
for k in WANT: byshard.setdefault(wm[k], []).append(k)

DT={"BF16":(np.uint16,2),"F32":(np.float32,4),"F16":(np.float16,2)}
out={}
for shard, keys in byshard.items():
    url=BASE+shard
    n=struct.unpack("<Q", get(url,(0,7)))[0]
    hdr=json.loads(get(url,(8,8+n-1)))
    print(f"{shard}: header {n} bytes, {len(hdr)-1} tensors")
    for k in keys:
        m=hdr[k]; s,e=m["data_offsets"]; dt,sz=DT[m["dtype"]]
        raw=get(url,(8+n+s, 8+n+e-1))
        a=np.frombuffer(raw,dtype=dt)
        if m["dtype"]=="BF16":
            a=(a.astype(np.uint32)<<16).view(np.float32)
        out[k]=a.reshape(m["shape"]).copy()
        print(f"   {k.split('layers.0.')[1]:<42} {m['dtype']:<5} {m['shape']}  {(e-s)/1024:.1f} KiB")
np.savez("/tmp/gemma_router_layer0.npz", **{k.split("layers.0.")[1]:v for k,v in out.items()})
print("saved /tmp/gemma_router_layer0.npz")
