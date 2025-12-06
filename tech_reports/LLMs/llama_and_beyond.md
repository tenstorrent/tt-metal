# Llama & Beyond: How I Extended `llama3_70b_galaxy` Optimizations To Qwen3-32b

Author: Rico Zhu

## Contents
- [The point of this report](motivation)
- [Bringing up models in `tt-metal`](bring-up)
    - [Initial perf estimates](roofline)
    - [Huggingface op profile](hf-op-profile)
    - [Model architecting]()
- [The timeline](timeline)
- [What went wrong](uh-oh)
- [Things I would do differently next time](reflection)

## The point of this write-up
I was inspired to write this tech report after reading a series of what I call "learning journey"-type blogposts that I see are growing more and more popular (see [here](https://www.aleksagordic.com/blog/matmul) and [here](https://www.thonking.ai/p/strangely-matrix-multiplications) for some great examples of other reports written in this style). What I wanted to convey was not the specifics of _what_ I did bring up Qwen3-32b using all of the optimizations the models team here at Tenstorrent has developed for Llama3-70b, but rather _how_ I went about the process of bringing this model up and _why_ I made the decisions that I did. Hence the point is to illustrate the full journey of bringing up a top-perf model on Tenstorrent's Wormhole Galaxy hardware from the perspective of a single developer, so that someone else in a similar position as me (a new-grad hire fresh out of college whose only experience working with a deep learning accelerating is calling `.cuda()` in Pytorch) can treat this as an onboarding document.

In the process, I hope to expose some tribal knowledge thus far hidden to the team which if I had known prior to starting this project, would have saved me weeks of tinkering. Some of the work is specific to the developer environment for the Tenstorrent Wormhole Galaxy as of August - October 2025, so I leave it to the reader's discretion for whether each example listed below applies to them.

## Bringing up models in `tt-metal`
