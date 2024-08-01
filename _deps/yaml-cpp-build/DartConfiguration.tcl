# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: /localdev/smanoj/tt-metal/.cpmcache/yaml-cpp/905940a232aea3437e2fb77334461b1fe343318b
BuildDirectory: /localdev/smanoj/tt-metal/_deps/yaml-cpp-build

# Where to place the cost data store
CostDataFile:

# Site is something like machine.domain, i.e. pragmatic.crd
Site: blr-t3k-01

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: Linux-clang++-17

# Subprojects
LabelsForSubprojects:

# Submission information
SubmitURL: http://

# Dashboard start time
NightlyStartTime: 00:00:00 EDT

# Commands for the build/test/submit cycle
ConfigureCommand: "/usr/bin/cmake" "/localdev/smanoj/tt-metal/.cpmcache/yaml-cpp/905940a232aea3437e2fb77334461b1fe343318b"
MakeCommand: /usr/bin/cmake --build . --config "${CTEST_CONFIGURATION_TYPE}"
DefaultCTestConfigurationType: Release

# version control
UpdateVersionOnly:

# CVS options
# Default is "-d -P -A"
CVSCommand: CVSCOMMAND-NOTFOUND
CVSUpdateOptions: -d -A -P

# Subversion options
SVNCommand: SVNCOMMAND-NOTFOUND
SVNOptions:
SVNUpdateOptions:

# Git options
GITCommand: /usr/bin/git
GITInitSubmodules:
GITUpdateOptions:
GITUpdateCustom:

# Perforce options
P4Command: P4COMMAND-NOTFOUND
P4Client:
P4Options:
P4UpdateOptions:
P4UpdateCustom:

# Generic update command
UpdateCommand: /usr/bin/git
UpdateOptions:
UpdateType: git

# Compiler info
Compiler: /usr/bin/clang++-17
CompilerVersion: 17.0.6

# Dynamic analysis (MemCheck)
PurifyCommand:
ValgrindCommand:
ValgrindCommandOptions:
MemoryCheckType:
MemoryCheckSanitizerOptions:
MemoryCheckCommand: MEMORYCHECK_COMMAND-NOTFOUND
MemoryCheckCommandOptions:
MemoryCheckSuppressionFile:

# Coverage
CoverageCommand: /usr/bin/gcov
CoverageExtraFlags: -l

# Cluster commands
SlurmBatchCommand: /usr/bin/sbatch
SlurmRunCommand: /usr/bin/srun

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summarily terminated.
# Currently set to 25 minutes
TimeOut: 1500

# During parallel testing CTest will not start a new test if doing
# so would cause the system load to exceed this value.
TestLoad:

UseLaunchers:
CurlOptions:
# warning, if you add new options here that have to do with submit,
# you have to update cmCTestSubmitCommand.cxx

# For CTest submissions that timeout, these options
# specify behavior for retrying the submission
CTestSubmitRetryDelay: 5
CTestSubmitRetryCount: 3
