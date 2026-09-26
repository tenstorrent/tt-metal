# Datatype sweep

Configs: `performance` = FF1/FF3+FF2 bfp4 (LoFi), attention/KV bfp8; `accuracy` = FF2 bfp8 (HiFi2), attention bfp8 HiFi2.
Token accuracy vs the HF bf16 refpt (tale-of-two-cities, 512-token teacher-forced continuation), batch-1 decode t/s/u.

```
{
 "p150": {
  "performance": {
   "top1": 95.0,
   "top5": 99.8,
   "tok_s_user": 12.18,
   "ttft_ms": 460.97
  },
  "accuracy": {
   "top1": 96.8,
   "top5": 99.8,
   "tok_s_user": 10.41,
   "ttft_ms": 485.45,
   "error": null
  }
 },
 "p300": {
  "performance": {
   "top1": 95.8,
   "top5": 99.8,
   "tok_s_user": 16.77,
   "ttft_ms": 299.77
  },
  "accuracy": {
   "top1": 97.2,
   "top5": 99.8,
   "tok_s_user": 15.01,
   "ttft_ms": 316.25,
   "error": null
  }
 },
 "p300x2": {
  "performance": {
   "top1": 95.2,
   "top5": 99.8,
   "tok_s_user": 25.17,
   "ttft_ms": 177.05
  },
  "accuracy": {
   "top1": 97.2,
   "top5": 99.8,
   "tok_s_user": 23.1,
   "ttft_ms": 179.54,
   "error": null
  }
 }
}
```

Selection per profile:
```
{
 "p150": {
  "optimizations": "accuracy",
  "why": "top-1 +1.8 at 15% t/s/u cost"
 },
 "p300": {
  "optimizations": "accuracy",
  "why": "top-1 +1.4 at 10% t/s/u cost"
 },
 "p300x2": {
  "optimizations": "accuracy",
  "why": "top-1 +2.0 at 8% t/s/u cost"
 }
}
```
