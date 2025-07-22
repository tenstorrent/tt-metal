# Adding a model to `experimental`

- Any new model should be added to `experimental`.
- Follow the steps to [graduate `experimental` model to a `demo`](MODEL_GRADUATION.md).

# A recommended dev flow on GitHub for adding new models

One common development flow that we see at Tenstorrent when adding a completely
new model is to have giant features changes in one PR, and try to run a whole
gamut of pipelines that cover all conceivable kinds of tests - end-to-end
performance tests, on-device performance tests, sub-model tests, demo tests
etc.

Because these kinds of pipelines are not run via merge queue or push to main
because of resource constraints, the ground often moves underneath the feet
of these large PRs. This leads to:

- Delays merging into main
- Review strain for reviewers because there is so much code in one PR
- Redundant pipeline usage because one test could be validated but now you're
  just running the whole set of single card tests to validate one mistake

This is why these larger model PRs can take a long time to merge.

The solution? Break your PRs up. The recommended flow is the following:

- (1 PR) The core model code, documentation, and model-specific component tests
  - Only need to run post-commit and frequent ttnn + models nightly
- (1 PR) The performance tests.
  - Only need to run model and device perf, and you could use [Pipeline
    select](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml).
    Note that you must select to build with Tracy if you run the device perf
    pipeline.
- (1 PR) The demo test.
  - Only need to run demo tests.

Note that the [Pipeline
select](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml)
can run all single-card pipelines above, except for post-commit. Note that
there should be little need to run this pipeline with all options on if you
follow the workflow above.

So the total number of pipelines you run is the same, but you only need to run
the specific pipeline(s) at each PR step.

Why do they all have to be together? Make your life easier and reduce the
ground that could shift beneath you.
