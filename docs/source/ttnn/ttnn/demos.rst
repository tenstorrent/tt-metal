Building and Uplifting Demos
############################

Working on models is incredibly exciting where exploratory work is done under the models/experimental folders
allowing the freedom to try new ideas and improve performance using Tenstorring hardware.   When a model is ready
to be showcased, we have a few important questions to answer before it is moved to our models/demos folder.
What you see below highlights the expectations for all our models that successfully migrate into the models/demos folder.

1. Does your model have good documenation?
    - We expect a ``README.md`` file within your model's folder that describes what the model is doing and how to run the model. This documentation should also include:
        - The origin of the model and credit to the originating authors, as appropriate.
        - Examples of how to use the model, including any necessary installation steps or dependencies.
        - A section on troubleshooting common issues or errors users might encounter.

2. Does your model have a test demonstrating that you have adequately achieved acceptable values for the final pearson's correlation coefficient (pcc) and has this been integrated on our CI pipeline?
    - Your model should include unit tests that validate its performance metrics, specifically the PCC, to ensure it meets our standards for accuracy.
    - These tests should be integrated into our Continuous Integration (CI) pipeline to automatically run against your model upon every commit. This ensures ongoing compliance and catches any regressions early.
    - Please include documentation in your ``README.md`` on how to execute these tests locally.
    - The PCC check should be included as part of the device and end-to-end performance that will be collected by our CI pipeline when running run_performance.sh.

3. What are the device performance metrics and have you integrated this on our CI pipeline?
    - Document the expected device performance metrics, including but not limited to inference time and memory usage, under a variety of conditions and configurations.
    - Include scripts or instructions to measure these metrics in the ``README.md``. Ensure these scripts are also part of the CI pipeline to automatically validate the device performance.
    - Update the documentation with any special hardware requirements or configurations needed to achieve these performance metrics.
    - Please add your test to the function run_device_perf_models() whithin run_performance.sh to collect and report on device metrics.  Only tests that have been marked with the annotation @pytest.mark.models_device_performance_bare_metal will be scheduled to run.

4. What are the end-to-end performance metrics and have you integrated this on our CI pipeline?
    - End-to-end performance metrics should include the time from input preparation to output readiness and any other relevant metrics that apply to the model's use case.
    - Provide clear instructions or scripts to measure these end-to-end metrics within the ``README.md``. These should also be incorporated into the CI pipeline for automated testing.
    - Discuss any external dependencies or services required for the full operation of your model and how they impact the end-to-end performance.
    - Please add your tests to either run_perf_models_other(), run_perf_models_llm_javelin(), or run_perf_models_cnn_javelin() within run_performance.sh.  Only tests marked with @pytest.mark.models_performance_bare_metal or @pytest.mark.models_performance_virtual_machine will be included.


For the purpose of organizing your tests to be managed for collecting both device and end-to-end performance, please take a look at test_ttnn_functional_resnet50.py and how the ResNet50TestInfra class was setup.
