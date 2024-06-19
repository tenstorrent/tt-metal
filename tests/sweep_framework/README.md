# Sweep Framework

**WIP**

## Test Vector Generation

The test vector generator takes in lists of parameters from each op test file and generates all permutations of these parameters, and stores them as a batch in the test vector database.

**TODO:** Add configurability/constraints for which sets of parameters will be generate into a batch. For example, certain memory configurations, data types, etc. This will allow for more focussed batches which can be executed by the runner for certain edge cases, or more specific testing.

Run example:
`python tests/sweep_framework/parameter_generator.py --module-name add --output-dir vectors --arch grayskull`

## Test Runner

The test runner reads in test vectors from the vector database and executes the tests sequentially by calling the op test's run function with the specified vectors.
Test vectors can be selected by module, or batch id to run more specific sets of tests.

**Current Problems:**
- Typing of ttnn pybinded types. We need to determine how to serialize/deserialize ttnn/tt_lib objects and store them in test vectors to be executed later, and in the results. We would like to utilize SQL filtering to select tests by different configuration criteria, and be able to see which subsets are passing/failing/being executed.

Run example:
`python tests/sweep_framework/runner.py --module-name add --batch-id nSELPbCptZgP78GffMys5i --arch grayskull`

**TODO:** Results will be stored in a results database for future reporting and aggregation. Also need to add performance measurements and failure classification (hangs, pcc, etc.).

## Op Test Library

Test writers will add test files to the sweeps directory with possible parameters, and run functions.

## Database

Sqlite for now, postgres is the likely option for production implementation.
