#!/usr/bin/env bash

# Compute SFPI release version information.
# Emit as evaluable shell assignments or CMAKE script
# This is the source of truth as to how we determine arch and distro names.
# Canonical location is in tenstorrent/tt-sfpi project's script directory.

# One ring to rule them all,
# One ring to find them,
# One ring to bring them all
# and in the darkness bind them

# For the realm of mortals, using the toolchain
# eval local $($FILE SHELL [$pkg]) -- generate sfpi variables to install
# $FILE CMAKE [$pkg] -- emit cmake script to set variables

# For the realm of elves, releasing the toolchain
# eval $($FILE VERSION $version) -- generate sfpi variables for release
# $FILE CREATE [$dirs] -- emit sfpi-version file from hash files

# For the realm of dwarves, building the toolchain
# $FILE BUILD [$DIR]

set -e

if [[ ${#1} = 0 ]]; then
    cat >&2 <<EOF
Usage:
$0 MERGE \$FILES - Merge sfpi-version files
$0 DIST          - Emit system info
$0 HASH [\$PACKAGE] - name or hash a package
$0 BASE          - generate base package name
$0 SHELL [\$PKG] - shell use for PKG
$0 CMAKE [\$PKG] - CMAKE use for PKG
$0 BUILD [\$DIR] - clone and build a toolchain
EOF
    exit 1
fi

version_file="sfpi-version"

if [[ $1 = MERGE ]]; then
    vars=()
    exit_code=0
    shift
    for file in "$@"; do
	while read line; do
	    var=${line%%=*}
	    eval val=${line#*=}
	    found=false
	    for seen in ${vars[@]}; do
		if [[ $seen = $var ]]; then
		    found=true
		    break
		fi
	    done
	    if $found; then
		eval seen=\$$var
		if [[ $seen != $val ]]; then
		    echo "ERROR: Multiple values of $var" >&2
		    exit_code=1
		fi
	    else
		vars+=($var)
		eval $var="$val"
	    fi
	done <$file
    done
    echo '# sfpi version information'
    for var in ${vars[@]}; do
	eval val="\$$var"
	echo "$var='$val'"
    done
    exit $exit_code
fi

case $1 in
    HASH) source /dev/stdin ;;
    BASE) source /dev/stdin ;;
    DIST) source /dev/stdin ;;
    *) source $(dirname $0)/$version_file ;;
esac

# define host system
sfpi_dist=unknown
sfpi_pkg=
if [[ -r /etc/os-release ]]; then
    source /etc/os-release
    sfpi_dist=$ID
    # See if ID_LIKE indicates a debian or fedora clone
    for like in $ID_LIKE
    do
	case $like in
	    debian) sfpi_dist=debian; break;;
	    fedora) sfpi_dist=fedora; break;;
	esac
    done

    case $sfpi_dist in
	debian) sfpi_pkg=deb;;
	fedora) sfpi_pkg=rpm;;
	gentoo) sfpi_pkg=gentoo;;
    esac
fi
sfpi_arch=$(uname -m)


if [[ $1 != DIST ]]; then
    # define download location & name
    sfpi_url=$sfpi_repo/releases/download/
    sfpi_filename=sfpi_
    if [[ $1 == BASE ]]; then
	sfpi_filename+=$sfpi_base
	sfpi_url+=$sfpi_base
    else
	sfpi_filename+=$sfpi_version
	sfpi_url+=$sfpi_version
    fi
    sfpi_filename+=_${sfpi_arch}_${sfpi_dist}
fi

if [[ $1 == HASH ]]; then
    if [[ -n $2 ]]; then
	hash=$(${sfpi_hashtype}sum -b < $2)
	suffix=${2##*.}
	echo "sfpi_${sfpi_arch}_${sfpi_dist}_${suffix}_hash='${hash%% *}'"
	exit 0
    fi
elif [[ $1 != BASE ]] && [[ $1 != DIST ]]; then
    # querier of sfpi-version
    if [[ $1 = BUILD ]]; then
	sfpi_pkg=txz
    elif [[ -n $2 ]]; then
	sfpi_pkg=$2
    fi
    if [[ -n $sfpi_pkg ]]; then
       sfpi_filename+=".$sfpi_pkg"
    fi
    sfpi_hash=$(eval echo "\${sfpi_${sfpi_arch}_${sfpi_dist}_${sfpi_pkg}_hash-}")
fi

if [[ $1 = CMAKE ]]; then
    # CMake LIKES THINGS SHOUTED AT IT
    sfpi_HASHTYPE="${sfpi_hashtype^^}"
fi

if [[ $1 != BUILD ]]; then
    # now emit definitions
    for var in $(set -o posix ; set | sed -e '/^sfpi_/{s/=.*//;p}' -e d)
    do
	eval val="\$$var"
	# relies on no inserted quoting for meta-characters
	if [[ $1 = CMAKE ]]; then
	    echo "set(SFPI${var#sfpi} \"$val\")"
	else
	    echo "$var='$val'"
	fi
    done
    exit 0
fi

# Now clone and build into $2
src=$2
if [[ -z $src ]]; then
    src=$(pwd)/sfpi-src
fi

mkdir -p $src
cd $src

# duplicate to stderr if it's different to stdout
dupstderr () {
    if [[ $(readlink /dev/fd/1) != $(readlink /dev/fd/2) ]]; then
	tee /dev/fd/2
    else
	cat
    fi
}

dupstderr <<EOF
Building SFPI $sfpi_version
Working Directory: $src
EOF

check_missing_deps () {
    local missing=()
    case $sfpi_dist in
	gentoo)
	    # Use full cat/pkg atom only where the bare name is ambiguous
	    # (gcc: cross-*/gcc exists; python: many slots/impls).
	    # For everything else, match by name in /var/db/pkg so category
	    # migrations (e.g. sys-devel -> dev-build) don't break the check.
	    gentoo_has_pkg() {
		case $1 in
		    */*)
			portageq has_version / "$1" 2>/dev/null ;;
		    *)
			find /var/db/pkg -maxdepth 2 -name "$1-*" -quit 2>/dev/null \
			    | grep -q . ;;
		esac
	    }
	    local pkgs=(sys-devel/gcc autoconf automake bison dejagnu dev-tcltk/expect
		flex gawk patchutils dev-lang/python expat gmp dev-libs/mpc mpfr
		sys-apps/texinfo)
	    for pkg in "${pkgs[@]}"; do
		gentoo_has_pkg "$pkg" || missing+=("$pkg")
	    done
	    if [[ ${#missing[@]} -gt 0 ]]; then
		echo "Missing Gentoo packages; install with: emerge ${missing[*]}" >&2
	    fi
	    ;;
	debian)
	    local pkgs=(gcc g++ libexpat1-dev libgmp-dev libmpc-dev libmpfr-dev
		autoconf automake bison expect flex gawk patchutils python3 texinfo)
	    for pkg in "${pkgs[@]}"; do
		dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed' || missing+=("$pkg")
	    done
	    if [[ ${#missing[@]} -gt 0 ]]; then
		echo "Missing Debian packages; install with: apt-get install ${missing[*]}" >&2
	    fi
	    ;;
	fedora)
	    local pkgs=(gcc gcc-c++ expat-devel gmp-devel libmpc-devel mpfr-devel
		autoconf automake bison expect flex gawk patchutils python3 texinfo)
	    for pkg in "${pkgs[@]}"; do
		rpm -q "$pkg" &>/dev/null || missing+=("$pkg")
	    done
	    if [[ ${#missing[@]} -gt 0 ]]; then
		echo "Missing Fedora packages; install with: dnf install ${missing[*]}" >&2
	    fi
	    ;;
	*)
	    cat >&2 <<WARN

Install (or otherwise provide) the following components:
Common names: autoconf automake bison expect flex gawk patchutils python3 texinfo
Debian names: gcc g++ libexpat1-dev libgmp-dev libmpc-dev libmpfr-dev
Fedora names: gcc gcc-c++ expat-devel gmp-devel libmpc-devel mpfr-devel
Gentoo names: sys-devel/gcc dev-build/autoconf dev-build/automake sys-devel/bison dev-util/dejagnu dev-tcltk/expect sys-devel/flex sys-apps/gawk dev-util/patchutils dev-lang/python dev-libs/expat dev-libs/gmp dev-libs/mpc dev-libs/mpfr sys-apps/texinfo

This script cannot install them as it knows neither your system's
packaging system, nor how it might have named them. You will have to
research that from the above clues. If required components are missing
the build will fail, sometimes with a clueful message. Please report
any additional packages or issues you encounter by filing an issue at
https://github.com/tenstorrent/tt-metal/issues
WARN
	    ;;
    esac
    return ${#missing[@]}
}

deps_missing=0
check_missing_deps || deps_missing=$?

if ! [[ -d .git ]]; then
    if [[ -t 0 && $deps_missing -gt 0 ]]; then
	echo >&2
	read -p "Confirm you have read and understood the above:" yes
	if ! [[ $yes =~ ^[Yy] ]]; then
	    echo "Assuming you have anyway" >&2
	fi
    fi
    echo | dupstderr
    echo "Cloning the repository ..." | dupstderr
    (set -x
     git clone --depth 1 $sfpi_repo .)
fi
echo | dupstderr
echo "Fetching sfpi $sfpi_version ..." | dupstderr
(set -x
 git fetch --depth 1 origin "refs/tags/$sfpi_version:refs/tags/$sfpi_version"
 git fetch --depth 1 origin $sfpi_version
 git -c "advice.detachedHead=false" checkout $sfpi_version
 git submodule update --depth 1 --init --recursive)

echo | dupstderr
echo "Building ..." | dupstderr
(set -x; rm -rf build)
# GCC 16+ defaults to C++20 where u8"" literals become char8_t[], breaking
# libcody's S2C helper which expects char[]. Opt back out to pre-C++20 behaviour.
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-fno-char8_t"
(set -x; scripts/build.sh --test-tt 2>&1)

echo | dupstderr
echo "Packaging ..." | dupstderr
(set -x; scripts/release.sh --txz-only 1>&2)

cp build/release/$sfpi_filename ..

echo "SFPI build completed" | dupstderr
