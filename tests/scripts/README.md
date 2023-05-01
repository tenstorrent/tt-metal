# Regression scripts

# Code-specific topics

Topics covering code-specific structures or concepts in the regression scripts
will be described here. Explanations covering how to use, add, or modify to the
code will be placed here.

## Command-line arguments

Each regression script can add command-line arguments. This is in addition to
the common ones that are supplied for all scripts in the runtime. For example,
`--timeout` is a common argument supplied to all tests.

The main idea with command-line parameters is that we have two types of
parameters: **common** and **specific**.

**Common** parameters are added to the parser under `add_common_args_` function
in `scripts.cmdline_args`.

**Specific** parameters are added to the parser under
`add_test_type_specific_args_` function in `scripts.cmdline_args`. The
specific test suite type is provided as a enum key from the user, in this case
the regression script.

### Using command-line parameters in a regression script

To actually use params in the regression script, the user must
1) Instantiate the parser by calling `get_cmdline_args(TestSuiteType)`, where
`TestSuiteType` is a specific type as provided in `scripts.common`. Ex.
```
if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.LLRT)
```
If a test type is not supplied, this function will error out.

2) Call an argument extraction function, which will always be called
`get_*_arguments_from_cmdline_args`, where `*` is replaced with some name.
Usually, this will correspond to the test suite type. The arguments will
provided as a tuple that must be unpacked, with common arguments coming first,
then specific ones.
```
if __name__ == "__main__":
    cmdline_args = get_cmdline_args(TestSuiteType.LLRT)

    timeout, short_driver_tests, = get_llrt_arguments_from_cmdline_args(cmdline_args)
```

3) Now these arguments can be used anywhere you'd like.

Note: You cannot call `get_cmdline_args()` more than once. This is because we
want to enforce a singleton. Each regression script should be invoked
separately.

### Adding a specific command-line parameter

If you would like to add a new specific command-line parameter, there are a few
simple steps.

1) Tell the parser what to look for. If you are familiar with `argparse` in
Python, this is trivial. Under the function `add_test_type_specific_args_` in
`scripts.cmdline_args`, either add a new `TestSuiteType` handler or under
an existing one, add an `argparser` call with appropriate arguments. For this
example, let's say we want to add an argument for `num_devices` for
`LLRT` tests.
```
def add_test_type_specific_args_(argparser, test_suite_type=TestSuiteType.UNKNOWN):
    ...
    if test_suite_type == TestSuiteType.LLRT:
        ...
        argparser.add_argument("--num_devices", help="Use specified number of devices", dest="num_devices", type=int, default=1)
        ...
```
2) Under the appropriate lower-order function to extract parameters from an
args object, add the argument by accessing it within the tuple that's returned.
If you're creating a new test suite, then you will have to create a lower-order
function.
```
def get_llrt_specific_args_from_parsed_args_(parsed_args):
    return (
        parsed_args.num_processes,
        ...
        parsed_args.num_devices,
    )
```
