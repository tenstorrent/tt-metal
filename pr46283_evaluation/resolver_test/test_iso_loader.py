"""Does the from-scratch loader (written with every PyTorch implementation hidden)
actually compute the same model as our reference?

ground truth : run-1 adapter -> our hand-written reference  (verified PCC 1.00000000, max|d| 0)
under test   : run-2 from-scratch implementation, 1064 lines, no reference visible

Both are pointed at the same weights on disk.
"""
import importlib.util
import sys

import torch

torch.manual_seed(0)


def load(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pcc(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    if a.shape != b.shape:
        return float("nan"), float("nan"), f"SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a - b).abs().max().item()
    if a.std() == 0 or b.std() == 0:
        return float("nan"), d, "degenerate (zero variance)"
    c = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return c, d, ""


def report(name, a, b, thresh=0.99):
    c, d, note = pcc(a, b)
    ok = (not (c != c)) and c >= thresh
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} PCC {c:.8f}  max|d| {d:.3e}  {note}", flush=True)
    return ok


ISO = "/localdev/lserbedzija/resolver_test/demo_dir_iso/tests/pcc/_reference_loader.py"
ADP = "/localdev/lserbedzija/resolver_test/demo_dir/tests/pcc/_reference_loader.py"

print("=== loading ground truth (adapter -> our reference) ===", flush=True)
ref = load("adp", ADP).load_reference_model("/localdev/lserbedzija/resolver_test/voxtral-tts-native")
print(f"  {type(ref).__name__}  children={[k for k, _ in ref.named_children()]}", flush=True)
print(f"  public attrs: {sorted(a for a in dir(ref) if not a.startswith('_') and a not in dir(torch.nn.Module))}", flush=True)

print("=== loading model under test (from scratch) ===", flush=True)
iso = load("iso", ISO).load_reference_model("/localdev/lserbedzija/resolver_test/native_iso")
print(f"  {type(iso).__name__}  children={[k for k, _ in iso.named_children()]}", flush=True)
# it keeps the checkpoint's bf16; the ground truth runs the reference math in fp32, so match that
iso = iso.float()
print(f"  cast to fp32 for comparison (param dtype {next(iso.parameters()).dtype})", flush=True)

results = {}

# ---------------------------------------------------------------- 1. embeddings
print("\n=== 1. text embedding ===", flush=True)
ids = torch.tensor([[1, 4023, 918, 27, 5, 66120, 3, 981, 12, 7]])
with torch.no_grad():
    e_iso = iso.embed_tokens(ids)
    e_ref = ref.embed_text(ids)
    if e_ref.dim() == 2:
        e_ref = e_ref.unsqueeze(0)
results["embedding"] = report("embed_tokens", e_iso, e_ref)

# ---------------------------------------------------------------- 2. backbone
print("\n=== 2. backbone, 26 layers (RoPE / GQA / RMSNorm / SwiGLU) ===", flush=True)
embeds = e_ref.float()
with torch.no_grad():
    h_iso = iso.forward_backbone(embeds)
    h_ref = ref.forward(inputs_embeds=embeds)
if isinstance(h_ref, (tuple, list)):
    h_ref = h_ref[0]
if h_ref.dim() == 2:
    h_ref = h_ref.unsqueeze(0)
print(f"  iso {tuple(h_iso.shape)} mean {h_iso.mean():+.4f} std {h_iso.std():.4f}", flush=True)
print(f"  ref {tuple(h_ref.shape)} mean {h_ref.mean():+.4f} std {h_ref.std():.4f}", flush=True)
results["backbone"] = report("forward_backbone", h_iso, h_ref)

# ------------------------------------------- 3. its own KV cache vs recompute
print("\n=== 3. from-scratch KV cache: prefill + steps vs full recompute ===", flush=True)
with torch.no_grad():
    full = iso.forward_backbone(embeds)
    cache = iso.make_cache()
    pre = iso.forward_backbone(embeds[:, :6], positions=torch.arange(6), cache=cache)
    steps = [pre[:, -1:]]
    for t in range(6, 10):
        steps.append(iso.forward_backbone(embeds[:, t : t + 1], positions=torch.tensor([t]), cache=cache))
    inc = torch.cat([pre[:, :-1]] + steps, dim=1)
results["kv_cache"] = report("cached == recomputed", inc, full, thresh=0.999)

# ---------------------------------------------------------------- 4. codec
print("\n=== 4. audio codec decoder (codes -> 24 kHz waveform) ===", flush=True)
# Conventions differ and both are documented: our reference takes offset-stripped codes as
# (B, 37, T); the from-scratch loader's decode_audio takes model-emitted codes as (B, T, 37)
# and strips the 2 audio special tokens itself. Compare like with like on both entry points.
sem = torch.randint(0, 2048, (1, 1, 24))
ac = torch.randint(0, 21, (1, 36, 24))
stripped = torch.cat([sem, ac], dim=1)  # (B, 37, T), offset already stripped
with torch.no_grad():
    w_iso = iso.audio_tokenizer.decode(stripped)
    w_ref = ref.audio_tokenizer(stripped)
print(f"  iso wav {tuple(w_iso.shape)}  {w_iso.shape[-1]/24000:.2f}s  peak {w_iso.abs().max():.4f}", flush=True)
print(f"  ref wav {tuple(w_ref.shape)}  peak {w_ref.abs().max():.4f}", flush=True)
results["codec"] = report("codec decode (matched conv.)", w_iso.reshape(1, 1, -1), w_ref.reshape(1, 1, -1))

with torch.no_grad():  # and the model-emitted entry point, end to end
    emitted = stripped.transpose(1, 2) + 2  # (B, T, 37) as the backbone emits them
    results["codec_emitted"] = report(
        "decode_audio (model-emitted)",
        iso.decode_audio(emitted).reshape(1, 1, -1),
        ref.audio_tokenizer((emitted - 2).transpose(1, 2)).reshape(1, 1, -1),
    )

# ---------------------------------------------------------------- 5. flow
print("\n=== 5. flow matching (fixed x_0) -> 37 audio codes ===", flush=True)
llm_h = h_ref[:, -1].float()
x0 = torch.randn(1, 36, generator=torch.Generator().manual_seed(1234))
flow_ref = None
for attr in ("acoustic_transformer", "flow", "flow_matching"):
    if hasattr(ref, attr):
        flow_ref = getattr(ref, attr)
        print(f"  ground-truth flow attribute: ref.{attr}", flush=True)
        break

real_randn = torch.randn


def pinned_randn(*a, **kw):
    kw.pop("generator", None)
    out = real_randn(*a, **kw)
    if out.shape[-1] == x0.shape[-1]:
        return x0.reshape(out.shape).to(out.dtype)
    return out


with torch.no_grad():
    torch.randn = pinned_randn
    try:
        c_iso = iso.generate_audio_frame(llm_h)
    finally:
        torch.randn = real_randn
    c_ref = None
    if flow_ref is not None:
        for call in (
            lambda: flow_ref(llm_h, x_0=x0),
            lambda: flow_ref(llm_h, x0),
            lambda: flow_ref(llm_h),
        ):
            try:
                c_ref = call()
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        if c_ref is None:
            print(f"  ground-truth flow not callable as tried: {last}", flush=True)
print(f"  iso codes {tuple(c_iso.shape)} semantic={c_iso[0,0].item()} acoustic[:6]={c_iso[0,1:7].tolist()}", flush=True)
if c_ref is not None:
    if isinstance(c_ref, (tuple, list)):
        c_ref = c_ref[0]
    c_ref = c_ref.reshape(1, -1)
    print(f"  ref codes {tuple(c_ref.shape)} semantic={c_ref[0,0].item()} acoustic[:6]={c_ref[0,1:7].tolist()}", flush=True)
    same = int((c_iso.reshape(-1) == c_ref.reshape(-1)).sum())
    n = c_ref.numel()
    print(f"  {'PASS' if same == n else 'PARTIAL'}  exact code match {same}/{n}", flush=True)
    results["flow_codes_exact"] = same == n
    results["flow_semantic"] = c_iso[0, 0].item() == c_ref[0, 0].item()
else:
    print("  SKIP (could not reach ground-truth flow)", flush=True)

print("\n=== SUMMARY ===", flush=True)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print(f"\n{sum(results.values())}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results.values()) else 1)
