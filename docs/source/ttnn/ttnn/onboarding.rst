Onboarding New Functionality
############################

TTNN is intended to be a well documented and a clear API that maintains stability, reliability, maintainability all while simultaneously allowing the user to fine tune the performance of the operations themselves.
To achieve this goal, we ask that new functionality to the API follows the Test-Driven Development originally popularized by Kent Beck.  By following this approach, the expectation
is that the long term benefits will help us maintain our objectives. Please follow all the following steps when adding new functionality.

1. Submit a request for a Single Operation as an `Issue <https://github.com/tenstorrent/tt-metal/issues>`_ and select `For external users - Propose a feature`
    * Provide a clear description of its intended purpose. (`Example <https://github.com/tenstorrent/tt-metal/issues/4730>`_)
    * Add the label ttnn to the issue
    * Add a python reference implementation that is fully functional.  This reference implementation will be called the `fallback` implementation.
2. Create a branch that defines the API and references the issue in step 1.
    * When creating the branch, please follow the pattern of 'TTNN-<Issue Number>-<brief description>'.  For example, if the issue is 4730, the branch name would be `TTNN-4730-concat-operation`
    * Use the `fallback` reference implementation for the operation and implement the functionality.
    * Add the documentation in the rst format for the operation under `ttnn documentation <https://github.com/tenstorrent/tt-metal/tree/main/docs/source/ttnn/ttnn>`_
    * Add :ref:`sweep tests<ttnn.sweep_tests>` to the branch using the fallback implementation under `ttnn sweep tests <https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/sweep_tests/sweeps>`_
3. Update the issue referencing the pull requests after verifying that all the sweep tests run as expected.  A TTNN CODEOWNERS will review the PR and verify that the API is acceptable and that the sweep tests reflect the intended functionality.
4. If the pull request (PR) is accepted it will be merge into the main branch and a new branch should be created that adds the implementation.
    * The fallback implementation for the Operation should be left and will continue to be used for op-by-op PCC comparisons when debugging models (see `--ttnn-enable-debug-decorator`).
    * Create a new PR with the final implementation and add a comment to the issue.
5. If all the sweep tests with the intended implementation passes and the CODEOWNERS accept the code, it will be merged into main.
