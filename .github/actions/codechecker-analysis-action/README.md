# [CodeChecker](http://github.com/Ericsson/CodeChecker/) C++ Static Analysis action

GitHub Action to execute static analysis over using [CodeChecker](http://github.com/Ericsson/CodeChecker/) as its driver.
For C-family projects (C, C++, Objective-C, CUDA, etc.), CodeChecker supports driving the static analysis programs of [Clang](http://clang.llvm.org).
Several other static analysers' output can be integrated into CodeChecker through the [report converter](http://codechecker.readthedocs.io/en/latest/tools/report-converter/).

## Overview (for C-family projects)

⚠️ **CAUTION! This action has been written with commands that target Ubuntu-based distributions!**

This single action composite script encompasses the following steps:

  1. Obtain a package of the LLVM Clang suite's analysers, and CodeChecker.
  2. _(Optional)_ Log the build commands to prepare for analysis.
  3. Execute the analysis.
  4. Show the analysis results in the CI log, and create HTML reports that can be uploaded as an artefact. (Uploading is to be done by the user!)
  5. _(Optional)_ Check for the current commit introducing new bug reports against a known state. (Good for pull requests!)
  6. _(Optional)_ Upload the results to a running _CodeChecker server_. (Good for the main project!)


ℹ️ **Note:** Static analysis can be a time-consuming process.
It's recommended that the static analysis step is not sequential with the rest of a CI execution, but either runs as its own job in a workflow, or a completely distinct workflow altogether.

Please ensure that your project is completely configured for a build before executing this action.

ℹ️ **Note:** Static analysers can rely on additional information that is optimised out in a true release build.
Hence, it's recommended to configure your project in a **`Debug`** configuration.

### Specifying the project to analyse

Add the job into your CI as follows.
The two versions are mutually exclusive &mdash; you either can give a compilation database, or you instruct CodeChecker to create one.

#### Projects that can generate a [JSON Compilation Database](http://clang.llvm.org/docs/JSONCompilationDatabase.html) and build cleanly (no generated code)

Some projects are trivial enough in their build configuration that no additional steps need to be taken after executing `configure.sh`, `cmake`, or similar tools.
If you are able to generate a _compilation database_ from your build system **without** running the build itself, you can save some time, and go to the analysis immediately.

You can specify the generated compilation database in the `logfile` variable

```yaml
job:
  steps:
    # Check YOUR project out!
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Prepare a build
    - name: "Prepare build"
      run: |
        mkdir -pv Build
        cd Build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    # Run the analysis
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        logfile: ${{ github.workspace }}/Build/compile_commands.json

    # Upload the results to the CI.
    - uses: actions/upload-artifact@v2
      with:
        name: "CodeChecker Bug Reports"
        path: ${{ steps.codechecker.outputs.result-html-dir }}
```

#### Projects that need to self-creating a *JSON Compilation Database* or require generated code

Other kinds of projects might rely heavily on _generated code_.
When looking at the source code of these projects **without** a build having been executed beforehand, they do not compile &mdash; as such, analysis cannot be executed either.

In this case, you will need to instruct CodeChecker to log a build (and spend time doing the build) just before analysis.

You can specify the build to execute in the `build-command` variable.

```yaml
job:
  steps:
    # Check YOUR project out!
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Prepare a build
    - name: "Prepare build"
      run: |
        mkdir -pv Build
        cd Build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

    # Run the analysis
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        build-command: "cd ${{ github.workspace }}/Build; cmake --build ."

    # Upload the results to the CI.
    - uses: actions/upload-artifact@v2
      with:
        name: "CodeChecker Bug Reports"
        path: ${{ steps.codechecker.outputs.result-html-dir }}
```

### Breaking the build if there are static analysis warnings

If requested, the _`warnings`_ output variable can be matched against to execute a step in the job which breaks the entire job if **any** static analysis warnings were emitted by the project.

ℹ️ **Note:** Due to static analysis being potentially noisy and the reports being unwieldy to fix, the default behaviour and recommendation is to only report the findings but do not break the entire CI.

To get the reports in a human-consumable form, they must be uploaded somewhere first, before the failure step fails the entire job!

```yaml
job:
  steps:
    # Check YOUR project out!
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Prepare a build
    - name: "Prepare build"
      run: |
        mkdir -pv Build
        cd Build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

    # Run the analysis
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        build-command: "cd ${{ github.workspace }}/Build; cmake --build ."

    # Upload the results to the CI.
    - uses: actions/upload-artifact@v2
      with:
        name: "CodeChecker Bug Reports"
        path: ${{ steps.codechecker.outputs.result-html-dir }}

    # Break the build if there are *ANY* warnings emitted by the analysers.
    - name: "Break build if CodeChecker reported any findings"
      if: ${{ steps.codechecker.outputs.warnings == 'true' }}
      run: exit 1
```

### Uploading results to a CodeChecker server

If your project hosts a CodeChecker server somewhere, the job can be configured
to automatically create or update a run.

```yaml
# It is recommended that storing only happens for PUSH events, and preferably
# only for long-term branches.
on:
  push:

job:
  steps:
    # Check YOUR project out!
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Prepare a build
    - name: "Prepare build"
      run: |
        mkdir -pv Build
        cd Build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

    # Run the analysis
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        build-command: "cd ${{ github.workspace }}/Build; cmake --build ."
        store: true
        store-url: 'http://example.com:8001/MyProject'
        store-username: ${{ secrets.CODECHECKER_STORE_USER }}
        store-password: ${{ secrets.CODECHECKER_STORE_PASSWORD }}
        # store-run-name: "custom run name to store against"
```

### Acting as a CI gate on pull requests

CodeChecker is capable of calculating the difference between two analyses.
If an analysis of the stable version of the project is stored (see above) to a server, a job for pull requests can be configured that automatically rejects a pull request if it tries to introduce _new_ analysis findings.

To get the reports in a human-consumable form, they must be uploaded somewhere first, before the failure step fails the entire job!

```yaml
on:
  pull_request:

runs:
  steps:
    # Check the pull request out! (In pull_request jobs, the checkout action
    # automatically downloads the "after-merge" state of the pull request if
    # there are no conflicts.)
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Prepare a build
    - name: "Prepare build"
      run: |
        mkdir -pv Build
        cd Build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

    # Run the analysis
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        build-command: "cd ${{ github.workspace }}/Build; cmake --build ."

        store: ${{ github.event_name == 'push' }}
        store-url: 'http://example.com:8001/MyProject'
        store-username: ${{ secrets.CODECHECKER_STORE_USER }}
        store-password: ${{ secrets.CODECHECKER_STORE_PASSWORD }}
        # Keep the names for 'store' and 'diff' in sync, or auto-generated!
        # diff-run-name: "custom run name to store with"

        diff: ${{ github.event_name == 'pull_request' }}
        diff-url: 'http://example.com:8001/MyProject'
        diff-username: ${{ secrets.CODECHECKER_DIFF_USER }}
        diff-password: ${{ secrets.CODECHECKER_DIFF_PASSWORD }}
        # diff-run-name: "custom run name to diff against"

    # Upload the potential new findings results to the CI.
    - uses: actions/upload-artifact@v2
      if: ${{ steps.codechecker.outputs.warnings-in-diff == 'true' }}
      with:
        name: "New introduced results Bug Reports"
        path: ${{ steps.codechecker.outputs.diff-html-dir }}

    - name: "Fail the job if new findings are introduced"
      if: ${{ steps.codechecker.outputs.warnings-in-diff == 'true' }}
      shell: bash
      run: |
        echo "::error title=New static analysis warnings::Analysed commit would introduce new static analysis warnings and potential bugs to the project"
        # Fail the build, after results were collected and uploaded.
        exit 1
```

## Overview (for other analyses through the _report-converter_)

⚠️ **CAUTION! This action has been written with commands that target Ubuntu-based distributions!**

This single action composite script encompasses the following steps:

  1. Obtain a package of CodeChecker.
  3. Use the `report-converter` to convert other analysers' reports to CodeChecker's format.
  4. Show the analysis results in the CI log, and create HTML reports that can be uploaded as an artefact. (Uploading is to be done by the user!)
  5. _(Optional)_ Check for the current commit introducing new bug reports against a known state. (Good for pull requests!)
  6. _(Optional)_ Upload the results to a running _CodeChecker server_. (Good for the main project!)


ℹ️ **Note:** Static analysis can be a time-consuming process.
It's recommended that the static analysis step is not sequential with the rest of a CI execution, but either runs as its own job in a workflow, or a completely distinct workflow altogether.

Please refer to the documentation of the analyser of your choice for this.
CodeChecker does **NOT** support driving the analysis through external tools, but if a successful analysis had been done, it can convert and store the results.

```yaml
job:
  steps:
    # Check YOUR project out!
    - name: "Check out repository"
      uses: actions/checkout@v2

    # Perform the analysis. Details vary between analysers!
    # Example for "PyLint" added below!
    - name: "Analyse with PyLint"
      run: |
        sudo apt-get -y install pylint
        pylint -f json --exit-zero myproject > pylint_reports.json

    # Run the conversion
    - uses: whisperity/codechecker-analysis-action@v1
      id: codechecker
      with:
        report-converter: true
        original-analyser: "pylint"
        original-analysis-output: "pylint_reports.json"

    # Upload the results (after conversion by CodeChecker) to the CI.
    - uses: actions/upload-artifact@v2
      with:
        name: "CodeChecker Bug Reports"
        path: ${{ steps.codechecker.outputs.result-html-dir }}
```

### Uploading results and acting as a CI gate

The _report-converter_ tool converts the output of various analysers to the common format used by CodeChecker.
Once the conversion is done, the rest of the action's features can execute in the same fashion as for C/C++ projects.
Please refer to earlier parts of the documentation for the configuration of these features.

## Action configuration

| Variable | Default                             | Description                                                                                                                                                                                                                                                                                                                                                  |
|----------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config` | `$(project-root)/.codechecker.json` | The configuration file containing flags to be appended to the analysis commands. It is recommended that most of the analysis configuration is versioned with the project. 🔖 Read more about the [`codechecker.json`](http://codechecker.readthedocs.io/en/latest/analyzer/user_guide/#configuration-file) configuration file in the official documentation. |

### Versions to install

| Variable         | Default                                                          | Description                                                                                                                                                                                                                                                                                                                                      |
|------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `llvm-version`   | `latest`                                                         | The major version of LLVM to install and use. LLVM is installed from the community [PPA](http://apt.llvm.org/). The value **MUST** be a major version (e.g. `13`) that is supported by the _PPA_ for the OS used! If `latest`, automatically gather the latest (yet unreleased) version. If `ignore`, don't install anything. (Not recommended.) |
| `install-custom` | `false`                                                          | If set to `true`, opens the ability to locally clone and install CodeChecker from the specified `repository` and `version`. Otherwise, `version` is taken as a release version, and the [CodeChecker suite from PyPI](http://pypi.org/project/codechecker) is downloaded.                                                                        |
| `repository`     | [`Ericsson/CodeChecker`](http://github.com/Ericsson/CodeChecker) | The CodeChecker repository to check out and build, if `install-custom` is `true`.                                                                                                                                                                                                                                                                |
| `version`        | `master`                                                         | If `install-custom` is `false`, the release version (e.g. `6.18.0`) to download from PyPI, or `master` to fetch the latest release. Otherwise, the branch (defaulting to `master`), tag, or commit SHA in the `repository` to check out.                                                                                                         |

### Build log configuration

🔖 Read more about [`CodeChecker log`](http://codechecker.readthedocs.io/en/latest/analyzer/user_guide/#log) in the official documentation.

| Variable        | Default | Description                                                                                                                                                                                                                          |
|-----------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `logfile`       |         | The location of the JSON Compilation Database which describes how the project is built. This flag is used if the build system can pre-generate the file for us.                                                                      |
| `build-command` |         | The build command to execute. CodeChecker is capable of executing and logging the build for itself. This flag is used if the build-system can not generate the information by itself, or the project relies on other generated code. |

### Analysis configuration

🔖 Read more about [`CodeChecker analyze`](http://codechecker.readthedocs.io/en/latest/analyzer/user_guide/#analyze) in the official documentation.

| Variable                 | Default          | Description                                                                                                                                                                                                                                                  |
|--------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `analyze-output`         | (auto-generated) | The directory where the **raw** analysis output should be stored.                                                                                                                                                                                            |
| `ctu`                    | `false`          | Enable [Cross Translation Unit analysis](http://clang.llvm.org/docs/analyzer/user-docs/CrossTranslationUnit.html) in the _Clang Static Analyzer_. ⚠️ **CAUTION!** _CTU_ analysis might take a very long time, and CTU is officially regarded as experimental. |
| `ignore-analyze-crashes` | `true`           | If set to `true`, the analysis phase will not report an error if some analysis actions fail (due to potential crashes in Clang).                                                                                                                             |

### Report configuration

🔖 Read more about [`CodeChecker parse`](http://codechecker.readthedocs.io/en/latest/analyzer/user_guide/#parse) in the official documentation.

### Report conversion configuration

🔖 Read more about the [`report-converter`](http://codechecker.readthedocs.io/en/latest/tools/report-converter/) in the official documentation.

| Variable                   | Default | Description                                                                                                                                        |
|----------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `report-converter`         | `false` | If set to `true`, the job will execute _report conversion_ from other analysers instead of driving the static analysis by itself.                  |
| `original-analyser`        |         | The "type" of the analysis that **had been performed** earlier. Passed as mandatory input to the `report-converter` executable.                    |
| `original-analysis-output` |         | The file or directory where the results of the third-party analyser are available. Passed as mandatory input to the `report-converter` executable. |

### Diff configuration

🔖 Read more about [`CodeChecker cmd diff`](http://codechecker.readthedocs.io/en/latest/analyzer/web_guide/#cmd-diff) in the official documentation.

🔓 Checking the analysis results against the contents of a server requires the `PRODUCT_VIEW` permission, if the server is requiring authentication.

| Variable        | Default                                                  | Description                                                                                                                                                                                                                                                                                                                     |
|-----------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `diff`          | `false`                                                  | If set to `true`, the job will compute a diff of the current analysis results against the results stored on a remote server.                                                                                                                                                                                                    |
| `diff-url`      |                                                          | The URL of the CodeChecker product to check and diff against, **including** the [endpoint](http://codechecker.readthedocs.io/en/latest/web/user_guide/#product_url-format). Usually in the format of `http://example.com/ProductName`. Specifying this variable is **required** if `diff` was set to `true`.                    |
| `diff-username` |                                                          | If the server requires authentication to access, specify the username which the check should log in with.                                                                                                                                                                                                                       |
| `diff-password` |                                                          | The password or [generated access token](http://codechecker.readthedocs.io/en/latest/web/authentication/#personal-access-token) corresponding to the user. 🔐 **Note:** It is recommended that this is configured as a repository secret, and given as such: `${{ secrets.CODECHECKER_PASSWORD }}` when configuring the action. |
| `diff-run-name` | (auto-generated, in the format `user/repo\: branchname`) | CodeChecker analysis executions are collected into _runs_. A run usually correlates to one configuration of the analysis.                                                                                                                                                                                                       |

### Store configuration

🔖 Read more about [`CodeChecker store`](http://codechecker.readthedocs.io/en/latest/web/user_guide/#store) in the official documentation.

🔓 Storing runs to a server requires the `PRODUCT_STORE` permission, if the server is requiring authentication.

| Variable         | Default                                                 | Description                                                                                                                                                                                                                                                                                                                     |
|------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `store`          | `false`                                                 | If set to `true`, the script will upload the findings to a CodeChecker server. Usually, other flags need to be configured too!                                                                                                                                                                                                  |
| `store-url`      |                                                         | The URL of the CodeChecker product to store to, **including** the [endpoint](http://codechecker.readthedocs.io/en/latest/web/user_guide/#product_url-format). Usually in the format of `http://example.com/ProductName`. Specifying this variable is **required** if `store` was set to `true`.                                 |
| `store-username` |                                                         | If the server requires authentication to access, specify the username which the upload should log in with.                                                                                                                                                                                                                      |
| `store-password` |                                                         | The password or [generated access token](http://codechecker.readthedocs.io/en/latest/web/authentication/#personal-access-token) corresponding to the user. 🔐 **Note:** It is recommended that this is configured as a repository secret, and given as such: `${{ secrets.CODECHECKER_PASSWORD }}` when configuring the action. |
| `store-run-name` | (auto-generated, in the format `user/repo: branchname`) | CodeChecker analysis executions are collected into _runs_. A run usually correlates to one configuration of the analysis. Runs can be stored incrementally, in which case CodeChecker is able to annotate that reports got fixed.                                                                                               |

## Action *`outputs`* to use in further steps

The action exposes the following outputs which may be used in a workflow's steps succeeding the analysis.

| Variable              | Value                                           | Description                                                                                                                          |
|-----------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `analyze-output`      | Auto-generated, or `analyze-output` input       | The directory where the **raw** analysis output files (either created by the analysers, or by the converter) are available.          |
| `codechecker-version` | Auto-generated (likely same as input `version`) | The version of the installed CodeChecker that performed the analysis.                                                                |
| `codechecker-hash`    | Auto-generated.                                 | The Git hash of the installed CodeChecker that performed the analysis.                                                               |
| `logfile`             | Auto-generated, or `logfile` input              | The JSON Compilation Database of the analysis that was executed.                                                                     |
| `llvm-version`        | Auto-generated.                                 | The full version string of the installed *LLVM/Clang* package (as reported by `clang --version`).                                    |
| `diff-html-dir`       | Auto-generated.                                 | The directory where the **user-friendly HTML** bug reports were generated to about the **new** findings (if `diff` was enabled).     |
| `diff-result-log`     | Auto-generated.                                 | `CodeChecker cmd diff`'s output log file which contains the **new** findings dumped into it.                                         |
| `diff-run-name`       | Auto-generated, or `diff-run-name` input        | The name of the analysis run (if `diff` was enabled) against which the reports were compared.                                        |
| `result-html-dir`     | Auto-generated.                                 | The directory where the **user-friendly HTML** bug reports were generated to.                                                        |
| `result-log`          | Auto-generated.                                 | `CodeChecker parse`'s output log file which contains the findings dumped into it.                                                    |
| `store-run-name`      | Auto-generated, or `store-run-name` input       | The name of the analysis run (if `store` was enabled) to which the results were uploaded to.                                         |
| `store-successful`    | `true` or `false`                               | Whether storing the results succeeded. Useful for optionally breaking the build later to detect networking failures.                 |
| `warnings`            | `true` or `false`                               | Whether the static analysers reported any findings.                                                                                  |
| `warnings-in-diff`    | `true` or `false`                               | If `diff` was enabled, whether there were **new** findings in the current analysis when compared against the contents of the server. |
