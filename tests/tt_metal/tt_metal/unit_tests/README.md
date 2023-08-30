# Summary
Unit testing uses the doctest framework.  See https://github.com/doctest/doctest/
Generally, there are three main levels of organization:
*  TEST_SUITE - Used to group main areas of tests
*  TEST_CASE - How Test case and sub-case gets split up is at test-writer discretion, but see the test_case section
*  SUB_CASE


## Build && Execution
### Build
`make tests/tt_metal/unit_tests`
### Get Help
`./build/test/tt_metal/unit_tests --help`
### Execute all tests
`./build/test/tt_metal/unit_tests`
### Execute filtered test-suite
`./build/test/tt_metal/unit_tests -ts="*Sfpu*"`
### List all test-suite with filter
`./build/test/tt_metal/unit_tests -ts="*Sfpu*" -lts`

## Folder Structure
General structure of the tests are as follows, more sub-folders can be added
<table><tr><td>
Directory Structure - Please add any new-tests to a corresponding folder.
</td></tr><td>
<pre lang="">
tt_metal/unit_tests/
&nbsp;&nbsp;> test_main.cpp
&nbsp;&nbsp;> basic/
&nbsp;&nbsp;&nbsp;&nbsp;> # Any basic test files can exist here, will be automatically added to test_main
&nbsp;&nbsp;> common/
&nbsp;&nbsp;&nbsp;&nbsp;> # Used to hold any common structures across all test suites like fixtures
&nbsp;&nbsp;> dram/
&nbsp;&nbsp;&nbsp;&nbsp;> # Any dram unit/stress test files can exist here, will be automatically added to test_main
&nbsp;&nbsp;> compute/
&nbsp;&nbsp;&nbsp;&nbsp;> # Any basic test files can exist here, will be automatically added to test_main
&nbsp;&nbsp;> new_folders/
&nbsp;&nbsp;&nbsp;&nbsp;> # Any test files can exist here, will be automatically added to test_main
test_utils/
&nbsp;&nbsp;> comparison.cpp # Useful utils for comparing, see example usages in unit tests
&nbsp;&nbsp;> print_helpers.cpp # Useful utils for printin
&nbsp;&nbsp;> stimulus.cpp # Useful utils for generating random vectors or specific vectors, see example usages in unit tests
&nbsp;&nbsp;> tilization.cpp # Useful utils for converting between tiled vectors or not, see example usages in unit tests
</td></tr></table>
