# TT-DiT Tests

Notes about TT-DiT (diffusion transformer) tests. **PLEASE UPDATE THIS** if anything is wrong or out of date! These notes are intended to give people some context/understanding about these tests, which seem out of place if you don't know why it's here.

TT-DiT code lives in `tt-metal/models/tt_dit`.  Test code for tt_dit lives in `tt-metal/models/tt_dit/tests`.

## Background
TT-DiT should be viewed as an independent arm within the models directory. It has it's own layers, it's own pipelines, some of it's own ops etc.

Given the development of TT-DiT has been somewhat independent of the development of the parent `models`, naturally the tests in `tt_dit/tests` do not necessarily reflect the same principles or strategy that `models` had.

## Relation to Model CI tests
We have attempted to integrate TT-DiT models into the Models CI testing framework described in `model_ci_tiers.md`.

However, there are components which don't neatly fit into the Model CI framework right now. See the section below for more information.

## IMPORTANT: Current Directory Structure
The directory/file structure in `tt_dit/tests` does not follow the same directory/file structure in `tt_dit` itself.

This difference between test structure and repo structure reflects my attempt to bridge the development style of TT-DiT against the need for standardizing model testing in Tenstorrent.

The curent structure is as follows:

- `models/` - All model tests. These contain the main tests used in the Model CI.
- `unit/` - Unit tests which are not model specific. These are an unfortunate edge case in the Models CI testing framework; we put all unit tests into a `TT-DiT common unit tests` module, which gets treated as by the Model CI as a "model", even though it actually isn't.
- `encoders/` - SHOULD BE added to CI, but currently isn't.
- `dataset_eval/` - NOT RUN IN CI. Only used for local development/evaluation of models!

The intent of the current structure is to make it simple to integrate a TT-DiT model into the Models CI testing framework.

For example when adding a new Model test to the Models CI framework, you would add a subfolder to `models` e.g. for Flux 1 there is the subfolder `models/tt_dit/tests/models/flux1/`, and place all relevant tests for that model (layers/pipelines/attention etc.) into that directory.

At the top level testing YAML, you can then declare the Flux 1 tests command to be simply `pytest models/tt_dit/tests/models/flux1/`. This avoids having to declare Flux 1 tests as a long list of different `.py` files that each pull the component you care about for said model (which you would be stuck with doing if the test directory was required to be a 1-1 match with the repo).
