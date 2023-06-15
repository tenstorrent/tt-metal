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
Director Structure
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

## Excerpt of how Test cases and subcases work 
Copied from doctest markdown https://github.com/doctest/doctest/blob/master/doc/markdown/tutorial.md#test-cases-and-subcases

Most test frameworks have a class-based fixture mechanism - test cases map to methods on a class and common setup and teardown can be performed in ```setup()``` and ```teardown()``` methods (or constructor/ destructor in languages like C++ that support deterministic destruction).

While **doctest** fully supports this way of working there are a few problems with the approach. In particular the way your code must be split up and the blunt granularity of it may cause problems. You can only have one setup/ teardown pair across a set of methods but sometimes you want slightly different setup in each method or you may even want several levels of setup (a concept which we will clarify later on in this tutorial). It was [**problems like these**](http://jamesnewkirk.typepad.com/posts/2007/09/why-you-should-.html) that led James Newkirk who led the team that built NUnit to start again from scratch and build [**xUnit**](http://jamesnewkirk.typepad.com/posts/2007/09/announcing-xuni.html)).

**doctest** takes a different approach (to both NUnit and xUnit) that is a more natural fit for C++ and the C family of languages.

This is best explained through an example:

```c++
TEST_CASE("vectors can be sized and resized") {
    std::vector<int> v(5);

    REQUIRE(v.size() == 5);
    REQUIRE(v.capacity() >= 5);

    SUBCASE("adding to the vector increases its size") {
        v.push_back(1);

        CHECK(v.size() == 6);
        CHECK(v.capacity() >= 6);
    }
    SUBCASE("reserving increases just the capacity") {
        v.reserve(6);

        CHECK(v.size() == 5);
        CHECK(v.capacity() >= 6);
    }
}
```

For each ```SUBCASE()``` the ```TEST_CASE()``` is executed from the start - so as we enter each subcase we know that the size is 5 and the capacity is at least 5. We enforce those requirements with the ```REQUIRE()``` macros at the top level so we can be confident in them. If a ```CHECK()``` fails - the test is marked as failed but the execution continues - but if a ```REQUIRE()``` fails - execution of the test stops.

This works because the ```SUBCASE()``` macro contains an if statement that calls back into **doctest** to see if the subcase should be executed. One leaf subcase is executed on each run through a ```TEST_CASE()```. The other subcases are skipped. Next time the next subcase is executed and so on until no new subcases are encountered.

So far so good - this is already an improvement on the setup/teardown approach because now we see our setup code inline and use the stack. The power of subcases really shows when we start nesting them like in the example below:

<table><tr><td>
Code
</td><td>
Output
</td></tr><tr><td>
<pre lang="c++">
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
<br>
#include &lt;iostream&gt;
using namespace std;
<br>
TEST_CASE("lots of nested subcases") {
&nbsp;&nbsp;&nbsp;&nbsp;cout << endl << "root" << endl;
&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cout << "1" << endl;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "1.1" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") {   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cout << "2" << endl;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "2.1" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cout << "2.2" << endl;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cout << "2.2.1" << endl;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "2.2.1.1" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "2.2.1.2" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "2.3" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SUBCASE("") { cout << "2.4" << endl; }
&nbsp;&nbsp;&nbsp;&nbsp;}
}
</pre>
</td><td width="400">
<pre lang="">
root
1
1.1<br>
root
2
2.1<br>
root
2
2.2
2.2.1
2.2.1.1<br>
root
2
2.2
2.2.1
2.2.1.2<br>
root
2
2.3<br>
root
2
2.4
</pre>
</td></tr></table>

Subcases can be nested to an arbitrary depth (limited only by your stack size). Each leaf subcase (a subcase that contains no nested subcases) will be executed exactly once on a separate path of execution from any other leaf subcase (so no leaf subcase can interfere with another). A fatal failure in a parent subcase will prevent nested subcases from running - but then that's the idea.

Keep in mind that even though **doctest** is [**thread-safe**](faq.md#is-doctest-thread-aware) - using subcases has to be done only in the main test runner thread and all threads spawned in a subcase ought to be joined before the end of that subcase and no new subcases should be entered while other threads with doctest assertions in them are still running.

