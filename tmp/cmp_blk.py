import sys, torch


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


ref, cand = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for k in ("video", "audio"):
    a, b = ref[k], cand[k]
    d = (a - b).abs()
    print(
        f"{k}: exact={torch.equal(a, b)} pcc={pcc(a, b):.7f} maxabs={d.max().item():.4g} mean|ref|={a.abs().mean().item():.4g}"
    )
