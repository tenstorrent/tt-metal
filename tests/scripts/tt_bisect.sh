#!/bin/bash
set -euo pipefail

: << 'END'
This script is used to find the commit that broke a test.
Flags:
    -f | --file : test file to run, also the test that broke
    -g | --good : good commit to start bisect
    -b | --bad : bad commit to start bisect
    -p | --path : commit-ish to cherry-pick onto each commit before building
    -t | --timeout : timeout duration for one iteration of the test
Example:
    ./tests/scripts/tt_bisect.sh -f ./build/test/tt_metal/test_add_two_ints -b HEAD -g 1eb7930
If the test involves multiple words you have to do "test_file":
    ./tests/scripts/tt_bisect.sh -f "pytest $TT_METAL_HOME/models/demos/resnet/tests/test_resnet18.py" -b HEAD -g 1eb7930
    ./tests/scripts/tt_bisect.sh -f "python tests/scripts/run_tt_metal.py --dispatch-mode fast" -b HEAD -g HEAD~10
END

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

timeout_duration_iteration=30m  # default per-iteration timeout
patch=""
while getopts "f:g:b:t:p:" opt; do
    case $opt in
         f | file)
            test=$OPTARG
            ;;
         g | good)
            good_commit=$OPTARG
            ;;
         b | bad)
            bad_commit=$OPTARG
            ;;
         t | timeout)
            timeout_duration_iteration=$OPTARG
            ;;
         p | patch)
            patch=$OPTARG
            ;;
         \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if ([ -z "$test" ] || [ -z "$good_commit" ] || [ -z "$bad_commit" ]); then
    echo "Please specify a test file, good commit and bad commit"
    exit 1
fi

# Validate good_commit SHA
if git cat-file -e "$good_commit" 2>/dev/null; then
    echo "Good commit SHA is valid: $good_commit"
else
    echo "Invalid good commit SHA: $good_commit"
    exit 1
fi

# Validate bad_commit SHA
if git cat-file -e "$bad_commit" 2>/dev/null; then
    echo "Bad commit SHA is valid: $bad_commit"
else
    echo "Invalid bad commit SHA: $bad_commit"
    exit 1
fi

echo "Time to find who broke it :)"
echo "Good commit:" $good_commit
echo "Bad commit:" $bad_commit
if ([ ! -z "$patch" ]); then
    echo "Cherry-pick commit:" $patch
fi



echo "Current location: `pwd`"
echo "Current branch: `git rev-parse --abbrev-ref HEAD`"
echo "Current commit: `git rev-parse HEAD`"
echo "Current status:"
echo `git status`

echo "git bisect start with good commit $good_commit and bad commit $bad_commit"
git bisect start $bad_commit $good_commit --

found=false

while [[ "$found" = "false" ]]; do
   echo "::group::Building `git rev-parse HEAD`"
   if ([ ! -z "$patch" ]); then
      git cherry-pick $patch
   fi
   git submodule update --recursive

   if [ -f rust/Cargo.lock ]; then
      grep 'name = "tokenizers"' -A 1 rust/Cargo.lock
   fi


   build_rc=0

   rm -rf /tmp/ccache

   mkdir -p /tmp/ccache

   ccache -z

   ./build_metal.sh --build-all --debug || build_rc=$?
   echo "::endgroup::"

   if [[ $build_rc -ne 0 ]]; then
      echo "Build failed; skipping this commit"
      git bisect skip
      continue
   fi

   echo "::group::Testing `git rev-parse HEAD`"
   timeout_rc=1
   max_retries=3
   attempt=1
   while [ $attempt -le $max_retries ]; do
      echo "Test attempt $attempt for commit $(git rev-parse HEAD)"
      echo "Running test: $test"
      timeout "$timeout_duration_iteration" bash -c "$test"
      timeout_rc=$?
      if [ $timeout_rc -eq 0 ]; then
         break
      else
         echo "Test failed (exit code $timeout_rc), retrying..."
         attempt=$((attempt + 1))
      fi
   done
   echo "Exit code: $timeout_rc"

   if ([ ! -z "$patch" ]); then
      # Must reset HEAD or git bisect good/bad will retry the merge base and we'll be stuck in a loop
      git reset --hard HEAD^
   fi
   echo "::endgroup::"

   if [ $timeout_rc -eq 0 ]; then
      echo "Commit is good"
      increment=$(git bisect good)
      echo "${increment}"
      first_line=$(echo "${increment}" | head -n 1)
   elif [ $timeout_rc -eq 124 ]; then
      echo "Test has timed out, skipping this commit"
      git bisect skip
      continue
   else
      echo "Commit is bad"
      increment=$(git bisect bad)
      echo "${increment}"
      first_line=$(echo "${increment}" | head -n 1)
   fi

   if [[ $first_line == *"is the first bad commit"* ]]; then
      echo "FOUND IT!: " $first_line
      found=true
   fi
done



git bisect reset
