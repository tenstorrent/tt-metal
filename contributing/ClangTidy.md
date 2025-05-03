# Scan the repo for clang-tidy violations
This is done automatically in post-commit, or can be run manually via [Code analysis · Workflow runs · tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal/actions/workflows/code-analysis.yaml)

# Enable a check
Delete the line from [tt-metal/.clang-tidy](../.clang-tidy) that disables the check.  Check [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html) and see if another check is an alias for the same.  If so, then delete the other name, too.  Now fix all the diagnostics that are now flagged!

> We take the approach of “enable everything, then disable specific checks” so that we have a clear TODO list to work on, rather than the opposite that only tells us where we’re currently at.

# Fix the violations
## Automatic
If the check being enabled has FIX-ITs (see [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html) and note the Offers fixes column), then leverage it by doing the following:

Prepare a clang-tidy tree

```
cmake --preset clang-tidy-fix
cmake --build --preset clang-tidy-fix --target clean
cmake --build --preset clang-tidy-fix
```
Go home for the day.  This will take a long long time if you only have a dozen cores.

When it’s done, review the changes it made and commit.  Re-run the final build step until it reaches the end.

Perform a final clean && build-everything to ensure everything was fixed.  Some checks have automatic fixes only for a subset of what it is able to diagnose.  The remaining diagnostics will need to be addressed manually.

## Manual
If the check does not have FIX-ITs, then it’s a manual process.  Perform the same steps as above, but after each build, review the log and manually address each diagnostic.

# FAQ
## What sorts of things can Clang Tidy detect?
Many things.  Some categories of checks are performance, security, modernization, readability, recommended practices, and convensions.  For a full list see [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html).

## Can we turn on ALL the checks?
No.  Some checks are mutually exclusive.  Generally when a style or opinion is involved.  For example [modernize-use-trailing-return-type](https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-trailing-return-type.html)  helps to use a trailing return type.  While [fuchsia-trailing-return](https://clang.llvm.org/extra/clang-tidy/checks/fuchsia/trailing-return.html)  enforces that such syntax is NOT used.

## What gotchas are there with the automatic FIX-ITs?
* When a FIX-IT is attempted on a header file referenced by multiple TUs, the resulting race condition can butcher the header file.  Workaround: git revert the affected file(s) and then run ninja -j1 to build single-threaded for long enough to process the header exactly once before going back to parallel builds.  If this is chronic, then consider running ninja -j1 -k0 and check back after the weekend.

* Some checks have FIX-ITs only for a subset of what it is able to detect.  Just because a check says it has FIX-ITs, don’t assume that it’s able to fix everything.  eg: when fixing performance-unnecessary-value-param, it seemed to not attempt an auto-fix inside templates or lambdas.  Those had to be manually adjusted.

* Sometimes the automatic edit can leave the repo in an unbuildable state.  This was encountered when fixing performance-unnecessary-value-param and a function was defined in one TU, and forward declares and invoked in another TU with no shared definition between them.  Clang-tidy diagnosed and fixed the definition, but the forward declaration was untouched, leaving it dangling and causing an undefinted reference at link time.

## Why are we scanning during a build instead of just pointing run-clang-tidy at the compile_commands.json?
In compile_commands.json, all code is equal.  But in tt-metal, some code is 3rd party that we aren’t interested in scanning.  We are unable to fully control this with judicious use of no-op .clang-tidy files as some 3rd party libs have their own .clang-tidy files.  As a result we need the full control of a build from CMake to scan what we need and not what we don’t need.  With ccache the build overhead is negligible in a clang-tidy scan, so it’s okay.

## What else should I know?
* Some checks have options to tweak the behaviour.  Review the documentation for the check you’re enabling if you feel like it should behave a little differently than it does.

* One check in particular *requires* options.  [readability-identifier-naming](https://clang.llvm.org/extra/clang-tidy/checks/readability/identifier-naming.html) has an elaborate set of knobs to define a naming convention and can even adjust existing code to the defined convention, within some constraints.  But without specifying the naming conventions, the check itself will do nothing even if enabled.
