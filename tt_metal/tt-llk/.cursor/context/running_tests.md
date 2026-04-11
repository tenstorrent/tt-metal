<environment-setup description="Setting up environment to run tests">
    <command executionPath="tests" if="using LLK IRD" id=0>
        `./setup_testing_env.sh`
    </command>
</environment-setup>

<running-test>
    <optional-arguments description="It will collect coverage code data, insert after `pytest`">`--coverage`</optional-arguments>
    <command executionPath="tests/python_tests" description="Compiles all the variants with given [test_name]">
        `pytest --compile-producer -n 10 -x ./test_name.py`
    </command>
    <command executionPath="tests/python_tests" description="Running compiled variants with given [test_name]">
        `pytest --compile-consumer -x ./test_name.py`
    </command>
</running-test>

<instructions>
    <step id=0>Setup Environment</step>
    <step id=1>Proceed with running-test</step>
</instructions>
