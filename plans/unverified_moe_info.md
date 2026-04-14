- GLM 4
    - Hidden: 5120, MoE intermediate: 1536.
    - Routed: 160, shared: 1. K:8
    - K: 8

- GLM 5
    - Hidden: 6144, intermediate: 2048
    - 256 routed experts, 1 shared.

- Kimi k 2.5
    Mostly same as DS
    384 routed experts. 1 shared
    K:8

- Qwen 3.5
    - Hidden: 2048, intermediate: 512
    - 256 experts, 1 shared

- Qwen3
    - Hidden: 2048, intermediate: 768
    - Experts: 128

- DeepSeek OCR
    - Hidden: 4096, intermediate: 1407
    - Note: shared expert intermediate dim: 5632
    - 64 routed experts, 2 shared
    - K=6

- Mistral 3.2 Large
    - Hidden: 6144, intermediate: 2048
    - 128 experts, 1 shared
    - K=8

- Ling-MoE 1T
    - Hidden: 8192, intermediate: 1536
    - Note: shared expert intermediate: 12288
    - 256 routed, 2 shared
    - K=8

- Existing support we want to make sure we don't regress on:
    - DeepSeek v3
        - Hidden: 7168, intermediate: 2048
        - 256 routed experts, 1 shared
        - K: 8

    - GPT-OSS
        - Hidden: 2048, intermediate: 2048
        - 128 routed experts, no shared expert
        - K: 8
