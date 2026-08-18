# Prefill ttnn ramp up

## Intro

Prefill models are all written in ttnn. They are large sequence len, chunked to smaller (SP) on where each chunk fits on either galaxy or sub galaxy systems, that is you can have large models that are chunked on an SC4 by seq-len (SP4), and then TP4x8 within the galaxy or smaller that are TP4, SP8 or inverse.

A TP4 model can be worked on on a QB2 in 1x4.

## Bringup process
The sequence is usually:
- Unit tests
- attention
- MoE
- MLP (if needed, don’t think so)
- LMHead
- embedding
- Single layer
- Full model
- Functional model bringup
- Perf measurements

## Pref Tuning
Once you have a functional layer, perf tuning begins:
- tracy to profile and https://github.com/tenstorrent/tt-perf-report/ to analyze
- prefill is compute bound, the vast majority of your time should be spent in MMs and SPDA
- mistrall small (https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 ) has 2 interesting features, it’s best to find similar ttnn models to see which ops you can re-use
- Multi latent attention (MLA)
- Mixture of experts (MoE)

# Next steps
Learn what models exist in ttnn, where they are, how they are defined (big models running on QB2 or Galaxy, or smaller models)
Learn how to run one w/ tracy and analyze performance. Can start with something small. Teach team on Friday.
TLDR: to take a look into finding ttnn models, where they are defined, what exists, how to run them and how to analyze with tracy tool

## Other folks steps
Alina: look into ttnn models, learn what it will take to write an Attention unit test for Mistral Small 4's Attention block with PCC checking.
