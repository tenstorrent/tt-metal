#!/bin/sh
die () {
  echo "`basename $0`: $*" 2>&1
  exit
}
[ -f VERSION ] || die "could not find 'VERSION'"
VERSION="`cat VERSION`"
[ -f makefile ] || die "could not find 'makefile' (run './configure' first)"
CC="`sed -e '/^CC/!d;s,^CC=,,' makefile`"
CFLAGS="`sed -e '/^CFLAGS/!d;s,^CFLAGS=,,' makefile`"
case "$CC" in
  gcc*|clang*)
    CC="`echo $CC|awk '{print \$1}'`"
    CC="`$CC --version 2>/dev/null|head -1`"
    ;;
esac
cat <<EOF
#define COMPILER "$CC $CFLAGS"
EOF
if [ -d .git -a .git/config ] 
then
  GITID="`git show|head -1|awk '{print $2}'`"
cat <<EOF
#define GITID "$GITID"
EOF
else
cat <<EOF
#define GITID 0
EOF
fi
cat <<EOF
#define VERSION "$VERSION"
EOF
LC_TIME="en_US"
export LC_TIME
DATE="`date 2>/dev/null|sed -e 's,  *, ,g'`"
OS="`uname -srmn 2>/dev/null`"
BUILD="`echo $DATE $OS|sed -e 's,^ *,,' -e 's, *$,,'`"
cat << EOF
#define BUILD "$BUILD"
EOF
