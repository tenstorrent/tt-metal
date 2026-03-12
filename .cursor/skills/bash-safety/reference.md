# Bash Safety — Complete Rule Reference

Every rule includes a **BAD** (vulnerable/broken) example, a **GOOD**
(corrected) example, and a brief rationale. Rules are numbered for
cross-referencing from code reviews.

Sources:
- [BashPitfalls](https://mywiki.wooledge.org/BashPitfalls)
- [Shellharden](https://github.com/anordal/shellharden/blob/master/how_to_do_things_safely_in_bash.md)

---

## Static Analysis

### SA-1: Run ShellCheck with all optional checks

Always run ShellCheck before committing. Use `-o all` to enable every
optional check category.

```bash
shellcheck -o all script.sh
```

For CI pipelines:

```bash
shellcheck -o all -f gcc script.sh   # machine-readable output
shellcheck -o all -s bash script.sh  # explicit shell dialect
```

Fix all errors (SC-prefixed codes). Suppress only with an inline comment
explaining why:

```bash
# shellcheck disable=SC2059  # format string is trusted and intentional
printf "$fmt" "$val"
```

---

## Quoting

### Q-1: Always quote variable expansions

Unquoted `$var` undergoes word splitting on `$IFS` (whitespace by default)
and pathname expansion (globbing). This breaks on filenames with spaces,
`*`, `?`, or `[`.

```bash
# BAD
cp $file $target
echo $msg

# GOOD
cp -- "$file" "$target"
printf '%s\n' "$msg"
```

### Q-2: Always quote command substitutions

The output of `$(cmd)` is split and globbed just like a variable.

```bash
# BAD
dir=$(dirname $f)
cd $(dirname "$f")

# GOOD
dir="$(dirname -- "$f")"
cd -P -- "$(dirname -- "$f")"
```

Quotes nest correctly inside `$()`: the inner and outer pairs are independent.

### Q-3: Quote variables inside `[ ]` (test)

Without quotes, empty or multi-word values cause syntax errors.

```bash
# BAD – breaks if $foo is empty or has spaces
[ $foo = bar ]
[ -n $foo ]
[ -z $foo ]

# GOOD
[ "$foo" = bar ]
[ -n "$foo" ]
[ -z "$foo" ]
```

### Q-4: Don't quote the right-hand side of `=~` in `[[ ]]`

Quoting the RHS of `=~` turns it into a literal string match instead of
a regex. Store complex patterns in a variable instead.

```bash
# BAD – literal string comparison, not regex
[[ $foo =~ 'some.*pattern' ]]

# GOOD
re='some.*pattern'
[[ $foo =~ $re ]]
```

### Q-5: Quoting the RHS of `=` in `[[ ]]` matters

Unquoted RHS is a glob pattern. Quote it for literal comparison.

```bash
# BAD – $bar is treated as a glob pattern
[[ $foo = $bar ]]

# GOOD – literal string comparison
[[ $foo = "$bar" ]]
```

### Q-6: Tilde is not expanded inside quotes

`"~"` is the literal character `~`, not the home directory.

```bash
# BAD
echo "~/mydir"         # prints ~/mydir literally
export foo=~/bar       # may not expand tilde depending on shell

# GOOD
echo "$HOME/mydir"
foo=~/bar; export foo
export foo="$HOME/bar"
```

### Q-7: Single quotes block parameter expansion

```bash
# BAD – $foo is not expanded
sed 's/$foo/replacement/'

# GOOD – use double quotes (escape sed metacharacters as needed)
sed "s/$foo/replacement/"
```

---

## Arrays

### A-1: Use real arrays, not whitespace-delimited strings

Strings break on filenames with spaces. Arrays handle them correctly.

```bash
# BAD
files="a b c"
for f in $files; do rm "$f"; done

# GOOD
files=(a b c)
for f in "${files[@]}"; do rm -- "$f"; done
```

### A-2: Use `"$@"` for positional parameters

`$*` and `$@` (unquoted) both undergo word splitting. `"$@"` preserves
each argument as a separate word.

```bash
# BAD
for arg in $*; do …; done
for arg in $@; do …; done

# GOOD
for arg in "$@"; do …; done
for arg; do …; done            # equivalent shorthand
```

### A-3: Quote array expansions

```bash
# BAD
for f in ${files[@]}; do …; done

# GOOD
for f in "${files[@]}"; do …; done
```

### A-4: Quote array element when passing to `unset`

Without quotes, `a[0]` can be interpreted as a glob.

```bash
# BAD
unset a[0]

# GOOD
unset -v 'a[0]'
```

### A-5: Don't populate arrays with raw command substitution

The output undergoes word splitting and globbing.

```bash
# BAD
hosts=( $(aws ec2 describe-instances …) )

# GOOD – single-line output
read -ra hosts < <(aws ec2 describe-instances …)

# GOOD – multi-line output (bash 4+)
readarray -t hosts < <(aws ec2 describe-instances …)
```

---

## Conditionals & Tests

### C-1: `[` is a command — spaces are mandatory

```bash
# BAD
[bar="$foo"]
[ bar="$foo" ]
[bar = "$foo"]

# GOOD
[ bar = "$foo" ]
[ "$bar" = "$foo" ]
```

### C-2: Use `=` not `==` inside `[ ]`

`==` is a bashism inside `[`. It works in bash but is not POSIX.

```bash
# BAD (not portable)
[ "$foo" == bar ]

# GOOD
[ "$foo" = bar ]
[[ $foo == bar ]]    # fine inside [[ ]]
```

### C-3: Don't use `&&` or `||` inside `[ ]`

`[` is a command; `&&`/`||` are shell operators that terminate the command.

```bash
# BAD – parsed as two separate commands
[ "$a" = x && "$b" = y ]

# GOOD
[ "$a" = x ] && [ "$b" = y ]
[[ $a = x && $b = y ]]
```

### C-4: `if` takes a command, not brackets

`[` is not part of `if` syntax. Use commands directly:

```bash
# BAD
if [grep -q foo myfile]; then …; fi

# GOOD
if grep -q foo myfile; then …; fi
```

### C-5: `cmd1 && cmd2 || cmd3` is not `if/then/else`

If `cmd2` fails, `cmd3` also runs.

```bash
# BAD – both branches can execute
[[ -s $log ]] && echo "Errors found" || echo "Clean"

# GOOD
if [[ -s $log ]]; then
    echo "Errors found"
else
    echo "Clean"
fi
```

### C-6: Check for broken symlinks

`[[ -e "$f" ]]` follows symlinks and returns false for broken ones.

```bash
# BAD – misses broken symlinks
[[ -e "$f" ]]

# GOOD
[[ -e "$f" || -L "$f" ]]
```

---

## Loops & Iteration

### L-1: Never parse `ls` output

`ls` mangles filenames (whitespace, globs, special chars). Use globs.

```bash
# BAD
for f in $(ls *.mp3); do …; done
for f in $(ls); do …; done
for f in `ls`; do …; done

# GOOD
for f in ./*.mp3; do
    [ -e "$f" ] || continue
    …
done
```

### L-2: Don't iterate `find` output with `for`

```bash
# BAD
for f in $(find . -type f); do …; done

# GOOD – find -exec
find . -type f -name '*.mp3' -exec some_command {} +

# GOOD – NUL-delimited read
while IFS= read -r -d '' f; do
    …
done < <(find . -type f -print0)

# GOOD – bash 4+ recursive glob
shopt -s globstar nullglob
for f in ./**/*.mp3; do …; done
```

### L-3: Don't use `for` to read lines from a file

```bash
# BAD – breaks on whitespace and globs
IFS=$'\n'
for line in $(cat file); do …; done

# GOOD
while IFS= read -r line; do
    …
done < file
```

### L-4: Pipe into while loop creates a subshell

Variable changes inside the loop are lost.

```bash
# BAD – count stays 0
count=0
grep foo bar | while read -r line; do ((count++)); done
echo "$count"   # always 0

# GOOD – process substitution
count=0
while read -r line; do
    ((count++))
done < <(grep foo bar)
echo "$count"
```

---

## Arithmetic

### AR-1: Use `(( ))` for integer comparison

`>` inside `[[ ]]` is string/collation comparison. Inside `[ ]` it's a
redirection.

```bash
# BAD
[[ $foo > 7 ]]      # string comparison
[ $foo > 7 ]         # creates file named "7"

# GOOD
(( foo > 7 ))
[ "$foo" -gt 7 ]     # POSIX
```

### AR-2: Brace expansion happens before parameter expansion

`{1..$n}` does not work because `$n` hasn't expanded yet.

```bash
# BAD
for i in {1..$n}; do …; done

# GOOD
for (( i = 1; i <= n; i++ )); do …; done
```

### AR-3: Validate input before arithmetic contexts

Arithmetic contexts (`(( ))`, `$(( ))`, `let`, array indices) evaluate
expressions, enabling code injection.

```bash
# BAD – user controls $num, can inject commands
read num
echo $(( num + 1 ))

# GOOD – validate first
if [[ $num =~ ^-?[0-9]+$ ]]; then
    echo $(( num + 1 ))
else
    echo "Invalid number" >&2; exit 1
fi
```

### AR-4: Beware of `(( i++ ))` exit status

`(( 0 ))` returns exit status 1 (false). This triggers `set -e`.

```bash
# BAD with set -e – exits if i is 0
i=0
(( i++ ))    # exit status 1 because expression value is 0 (pre-increment)

# GOOD – use pre-increment or suppress failure
(( ++i ))
(( i++ )) || true
```

---

## Command Substitution

### CS-1: Prefer `$(…)` over backticks

Backticks have confusing escaping rules when nested and are harder to read.

```bash
# BAD
dir=`dirname \`readlink -f "$f"\``

# GOOD
dir="$(dirname "$(readlink -f "$f")")"
```

### CS-2: Separate `local` from assignment with command substitution

`local` is a command whose exit status masks the substitution's exit status.

```bash
# BAD – exit status of cmd is lost
local var=$(cmd)

# GOOD
local var
var=$(cmd)
```

The same applies to `export` and `readonly`.

### CS-3: Command substitution strips trailing newlines

All trailing newlines are removed. If you must preserve them:

```bash
output="$(cmd; printf x)"
output="${output%x}"
```

### CS-4: errexit is ignored inside command arguments

```bash
# BAD – if nproc is missing, make gets -j with empty arg
set -e
make -j"$(nproc)"

# GOOD
set -e
jobs="$(nproc)"
make -j"$jobs"
```

---

## Redirections & Pipes

### R-1: Redirect order matters

`2>&1` duplicates stderr to wherever stdout is **currently** pointing.

```bash
# BAD – stderr goes to terminal, not logfile
cmd 2>&1 >>logfile

# GOOD – redirect stdout first, then dup stderr to it
cmd >>logfile 2>&1
```

### R-2: Don't read and write the same file in a pipeline

```bash
# BAD – truncates or fills disk
sed 's/foo/bar/' file > file

# GOOD
sed -i 's/foo/bar/' file                     # GNU sed
sed 's/foo/bar/' file > tmpfile && mv tmpfile file  # portable
```

### R-3: Don't close stdin/stdout/stderr

Programs assume fds 0-2 exist. Closing them causes unpredictable failures.

```bash
# BAD
cmd 2>&-

# GOOD
cmd 2>/dev/null
```

### R-4: `sudo cmd > /file` – redirect runs as current user

```bash
# BAD – redirect is not privileged
sudo cmd > /root/output

# GOOD
sudo sh -c 'cmd > /root/output'
cmd | sudo tee /root/output >/dev/null
```

---

## Filenames & Paths

### F-1: Handle filenames with leading dashes

A filename like `-rf` can be interpreted as an option.

```bash
# BAD
rm $file

# GOOD – double-dash ends option parsing
rm -- "$file"

# GOOD – prefix with ./
rm "./$file"
```

### F-2: Use `-print0` with `find` and `-d ''` with `read`

Filenames can contain newlines. NUL is the only byte forbidden in pathnames.

```bash
# GOOD
find . -name '*.log' -print0 | while IFS= read -r -d '' f; do
    rm -- "$f"
done
```

### F-3: Always check `cd` return status

Failing silently after `cd` can cause destructive commands in the wrong
directory.

```bash
# BAD
cd /foo; rm -rf ./*

# GOOD
cd /foo || exit 1
rm -rf ./*

# GOOD – subshell isolates directory change
(
    cd /foo || exit 1
    rm -rf ./*
)
```

### F-4: Don't export `CDPATH`

Exporting `CDPATH` changes `cd` behavior in child scripts, causing them
to silently change to the wrong directory.

```bash
# BAD
export CDPATH=.:~/projects

# GOOD – set but don't export
CDPATH=.:~/projects
```

### F-5: `sudo cmd /foo/*` – glob expands as current user

```bash
# BAD – glob runs unprivileged
sudo ls /foo/*

# GOOD
sudo sh -c 'ls /foo/*'
```

---

## Output & Echo

### O-1: Use `printf` instead of `echo`

GNU `echo` interprets leading `-n`, `-e`, `-E` as options and offers no
`--` to stop option parsing. This makes it impossible to safely print
arbitrary data.

```bash
# BAD – breaks if $var is "-n" or "-e flag"
echo "$var"

# GOOD
printf '%s\n' "$var"
```

### O-2: Never use a variable as `printf` format string

`%` and `\` in the variable are interpreted as format/escape directives.

```bash
# BAD – format string injection
printf "$user_input"

# GOOD
printf '%s' "$user_input"
```

### O-3: Use `cat <<EOF` or `printf` for multi-line text, not `echo <<EOF`

`echo` does not read from stdin.

```bash
# BAD
echo <<EOF
Hello
EOF

# GOOD
cat <<EOF
Hello
EOF

# GOOD
printf '%s\n' "Hello"
```

---

## Script Structure

### S-1: Use `#!/usr/bin/env bash`

More portable than `/bin/bash` (e.g. NixOS, Homebrew on macOS).

```bash
#!/usr/bin/env bash
```

Don't put options in the hashbang — they can be overridden by callers.

### S-2: Enable strict mode

```bash
set -euo pipefail
```

**Caveats to know:**

- **`errexit` (`-e`)**: Ignored inside command substitutions, functions
  called as conditions (`f && …`), and compound commands tested by `if`.
  Always add explicit `|| return` / `|| exit` checks in functions.
- **`nounset` (`-u`)**: In bash < 4.4, empty arrays are treated as unset.
  Use a feature check or skip `-u`:

```bash
if test "$BASH" = "" || "$BASH" -uc 'a=();true "${a[@]}"' 2>/dev/null; then
    set -euo pipefail
else
    set -eo pipefail
fi
```

- **`pipefail`**: An earlier stage exiting from SIGPIPE (e.g. `cmd | head`)
  makes the whole pipeline fail. Only enable for pipelines where every
  stage must succeed.

### S-3: Enable safer globbing

```bash
shopt -s nullglob globstar
```

- `nullglob`: unmatched globs expand to nothing instead of the literal
  pattern string.
- `globstar`: enables `**` for recursive matching.

### S-4: Assert command dependencies early

```bash
require() { hash "$@" || exit 127; }
require curl jq parallel
```

Prevents cryptic failures deep in the script.

### S-5: Use `function_name() {` not `function function_name()`

Combining the `function` keyword with `()` is not portable.

```bash
# BAD
function foo() { …; }

# GOOD
foo() { …; }
```

### S-6: End scripts with explicit exit status

The script's exit status is that of the last command. An accidental
conditional as the last command makes the script "fail" when it shouldn't.

```bash
# At end of script
exit 0
```

---

## Dangerous Patterns

### D-1: Don't use `eval` with untrusted data

```bash
# BAD
eval "$user_input"

# GOOD – use arrays for dynamic commands
cmd_args=("$program" --flag "$value")
"${cmd_args[@]}"
```

### D-2: Don't inject filenames into `sh -c` via `{}`

`find -exec sh -c 'echo {}'` is a code injection vulnerability. Pass
filenames as arguments.

```bash
# BAD
find . -exec sh -c 'echo {}' \;

# GOOD
find . -exec sh -c 'echo "$1"' _ {} \;
```

### D-3: Don't use arithmetic contexts with unsanitized input

Array subscripts and `$(( ))` evaluate expressions recursively.

```bash
# BAD – $x could be '$(rm -rf /)'
y=$(( array[$x] ))

# GOOD – validate, then use
[[ $x =~ ^[0-9]+$ ]] || exit 1
y=$(( array[x] ))
```

### D-4: Avoid `xargs` without `-0`

`xargs` splits on whitespace and interprets quotes by default.

```bash
# BAD
find . -name '*.txt' | xargs wc

# GOOD
find . -name '*.txt' -print0 | xargs -0 wc
find . -name '*.txt' -exec wc {} +     # no xargs needed
```

### D-5: Avoid non-atomic writes with `xargs -P`

Parallel `xargs` can interleave output from concurrent processes if writes
exceed the pipe buffer size (~8 KiB). Use GNU Parallel for serialized output,
or redirect each job to its own file.

---

## Miscellaneous

### M-1: Assignments have no spaces around `=`

```bash
# BAD
foo = bar    # runs "foo" with args "=" and "bar"
$foo=bar     # not how you assign
foo =bar     # runs "foo" with arg "=bar"

# GOOD
foo=bar
foo="bar"
```

### M-2: `su -c` needs an explicit username

```bash
# BAD
su -c 'some command'

# GOOD
su root -c 'some command'
```

### M-3: `&` is a command terminator — don't combine with `;`

```bash
# BAD
for i in {1..10}; do ./something &; done

# GOOD
for i in {1..10}; do ./something & done
```

### M-4: Call `date` only once

Multiple calls can span midnight, giving inconsistent values.

```bash
# BAD
month=$(date +%m)
day=$(date +%d)     # could be a different day

# GOOD
eval "$(date +'month=%m day=%d year=%Y')"
```

### M-5: `read $foo` — don't put `$` before the variable name

```bash
# BAD – reads into whatever variable $foo names
read $foo

# GOOD
IFS= read -r foo
```

### M-6: IFS as field terminator drops trailing empty fields

```bash
# BAD – trailing empty field is lost
IFS=, read -ra fields <<< "a,b,"

# GOOD – append separator to force scanning
input="a,b,"
IFS=, read -ra fields <<< "$input,"
```

### M-7: Forced base-10 conversion fails on negative numbers

```bash
# BAD – fails if $i is negative
i=$(( 10#$i ))

# GOOD
i=$(( ${i%%[!+-]*}10#${i#[-+]} ))
```

### M-8: Use `pgrep`/`pkill` instead of `ps | grep`

```bash
# BAD
pid=$(ps ax | grep '[m]yprocess' | awk '{print $1}')

# GOOD
pid=$(pgrep myprocess)
pkill myprocess
```

### M-9: Don't `set -e` and rely on `$?` after a command

With errexit, the script exits before `$?` can be inspected if the command
fails. Test the command directly instead.

```bash
# BAD
set -e
cmd
if [ $? -ne 0 ]; then …; fi   # never reached

# GOOD
if cmd; then
    …
else
    echo "cmd failed with $?" >&2
fi
```

### M-10: Use `while … done < <(foo)` not `while … done <<< "$(foo)"`

The here-string form collects all output first (no streaming), strips
trailing newlines, adds one back, and discards NUL bytes.

```bash
# BAD
while IFS= read -r line; do …; done <<< "$(cmd)"

# GOOD
while IFS= read -r line; do …; done < <(cmd)
```

### M-11: Cleanup with trap on EXIT

```bash
tmpfile="$(mktemp -t myprogram-XXXXXX)"
cleanup() { rm -f "$tmpfile"; }
trap cleanup EXIT
```

### M-12: Use `tr '[:upper:]' '[:lower:]'` not `tr [A-Z] [a-z]`

Unquoted brackets are globs; `A-Z` range depends on locale.

```bash
# BAD
tr [A-Z] [a-z]

# GOOD – locale-aware
tr '[:upper:]' '[:lower:]'

# GOOD – ASCII only
LC_COLLATE=C tr A-Z a-z
```
