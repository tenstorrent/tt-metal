# Llama & Beyond: How I Extended `llama3_70b_galaxy` Optimizations To Qwen3-32b

Author: Rico Zhu

## Contents
- [The point of this report](motivation)
- [Bringing up models in `tt-metal`](bring-up)
    - [Huggingface op profile](hf-op-profile)
    - [Initial perf estimates](roofline)
    - [Model architecting]()
- [The timeline](timeline)
- [What went wrong](uh-oh)
- [Things I would do differently next time](reflection)

## The point of this write-up
I was inspired to write this tech report after reading a series of what I call "learning journey"-type blogposts that I see are growing more and more popular (see [here](https://www.aleksagordic.com/blog/matmul) and [here](https://www.thonking.ai/p/strangely-matrix-multiplications) for some great examples of other reports written in this style). What I wanted to convey was not the specifics of _what_ I did bring up Qwen3-32b using all of the optimizations the models team here at Tenstorrent has developed for Llama3-70b, but rather _how_ I went about the process of bringing this model up and _why_ I made the decisions that I did. Hence the point is to illustrate the full journey of bringing up a top-perf model on Tenstorrent's Wormhole Galaxy hardware from the perspective of a single developer, so that someone else in a similar position as me (a new-grad hire fresh out of college whose only experience working with a deep learning accelerator is calling `.cuda()` in Pytorch) can treat this as an onboarding document.

In the process, I hope to expose some tribal knowledge thus far hidden to the models team which if I had known prior to starting this project, would have saved me weeks of tinkering. Some of the work is specific to the developer environment for the Tenstorrent Wormhole Galaxy as of August - October 2025, so I leave it to the reader's discretion whether each example listed below applies to them.

## Bringing up models in `tt-metal`
Internally within the models team, the process for bringing up a new model follows the rough outline below:
1. Functional bring-up (decode then prefill).
    - The point of this step is just to get something that _works_. Perf is not a top concern right now, the main point is to see if all of the ops present in the reference model exist in `ttnn`. If not, this is the time to communicate with the ops team and file the necessary Github issues for new features.
2. Optimizations.
    - Just getting models functional is not enough, and typically your naive implementation using just `ttnn` ops will get you anywhere from pretty bad perf to _really really_ bad perf (for instance when I first brought up Grok 2.5, the model ran at `<1 t/s/u`, which is objectively pretty awful).
    - Some optimizations are easier than others. For instance, enabling trace only involves taking out all reads and writes to DRAM within the scope of your trace call, which is usually pretty straightforward and can literally 10x your perf if you enable it. Focus on the easy wins first to stay on track.
3. vLLM integration.
    - The job of the models team is to deliver models running on Tenstorrent with top-perf to customers. Since most customers don't have the expertise to run your bare-metal `pytest`, we will have to connect our model to some sort of inference endpoint. At Tenstorrent, we have a fork of vLLM which serves this exact purpose. See `vLLM_integration.md` for more details.
    - If you have an actual customer waiting for you, it is at this point when you should contact someone on the customer team to help you set up `tt-inference-server` and run your model through the customer benchmarks.
4. Top-perf optimiations.
    - After getting a model customer ready (at least to a point where you won't be embarrased by its performance when doing a live demo), it is then time to get the model perf to reach the final target.

### 1. Functional bring-up
The process for functional bring-up can be boiled down to checking off all the boxes for the following checklist:
 - [ ] Get a high-level diagram of all the ops in your decode stage, including shapes. Create unit tests for each shape and op combination that occurs in your model, make sure that nothing hangs and PCC is good.
 - [ ] Creating classes for your main modules. For dense models these typically are `attention`, `mlp`, `decoder`. For MOE models we have `attention`, `gate`, `moe`, `decoder`.
 - [ ] Create module tests for each of the modules you implement above (e.g., `test_qwen_attention.py, test_qwen_mlp.py`), and ensure that nothing hangs and PCC is good.
 - [ ] Create your `model` and `generator` class to allow for single layer and full-model testing.
 - [ ] Once your full model passes, create a `demo_decode` script that lets you generate text at least doing "prefill by decode".
 - [ ] Do all of that again for prefill; create unit tests with a non-trivial sequence lenngth, create module tests for prefill (e.g., `test_qwen_attention_prefill.py, test_qwen_mlp_prefill.py`).
 - [ ] After prefill is done, create a `text_demo` script and then you should be ready to show off your full model!
