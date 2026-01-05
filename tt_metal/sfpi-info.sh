#! /bin/bash

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
$0 CREATE [\$DIRS] - generate release information
$0 VERSION \$VER - generate release names
$0 SHELL [\$PKG] - shell use for PKG
$0 CMAKE [\$PKG] - CMAKE use for PKG
$0 BUILD [\$DIR] - clone and build a toolchain
EOF
    exit 1
fi

version_file="sfpi-version"
hashtype=sha256
if [[ ${1-} = CREATE ]]; then
    # create sfpi-version file
    version=
    exit_code=0
    echo '# sfpi version information' >$version_file
    echo 'sfpi_repo=https://github.com/tenstorrent/sfpi' >>$version_file
    echo "sfpi_hashtype=$hashtype" >>$version_file
    shift
    if [[ ${#1} = 0 ]]; then
	set .
    fi
    files=()
    for dir in "$@"
    do
	for file in "$dir"/*.$hashtype
	do
	    if ! [[ -r "$file" ]]; then
		echo "$dir contains no $hashtype files" >&2
		exit 1
	    fi
	    files+=("$file")
	    ver="${file##*/sfpi_}"
	    ver="${ver%%_*}"
	    if [[ $ver != $version ]]; then
		if [[ -n $version ]]; then
		    echo "ERROR: Multiple versions" >&2
		    exit_code=1
		fi
		version=$ver
		echo sfpi_version=$version >>$version_file
	    fi
	done
    done
    sed 's/^\([0-9a-f]*\) \*sfpi_[^_]*_\([^.]*\)\.\(.*\)$/sfpi_\2_\3_hash=\1/' "${files[@]}" >>$version_file
    echo "$version_file for $version created"
    exit $exit_code
fi

if [[ ${1-} = VERSION ]]; then
    # releaser of sfpi
    sfpi_version=$2
    sfpi_hashtype=$hashtype
else
    source $(dirname $0)/$version_file
fi

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
    esac
fi
sfpi_arch=$(uname -m)

# define download location & name
sfpi_url=$sfpi_repo/releases/download/$sfpi_version
sfpi_filename=sfpi_${sfpi_version}_${sfpi_arch}_${sfpi_dist}

if [[ ${1-} != VERSION ]]; then
    # querier of sfpi-version
    if [[ ${1-} = BUILD ]]; then
	sfpi_pkg=txz
    elif [[ -n $2 ]]; then
	sfpi_pkg=$2
    fi
    sfpi_filename+=".$sfpi_pkg"
    sfpi_hash=$(eval echo "\${sfpi_${sfpi_arch}_${sfpi_dist}_${sfpi_pkg}_hash-}")
    unset sfpi_builton
fi
if [[ ${1-} = CMAKE ]]; then
    # CMake LIKES THINGS SHOUTED AT IT
    sfpi_HASHTYPE="${sfpi_hashtype^^}"
fi

if [[ ${1-} != BUILD ]]; then
    # now emit definitions
    for var in $(set -o posix ; set | sed -e '/^sfpi_/{s/=.*//;p}' -e d)
    do
	eval val="\$$var"
	# relies on no inserted quoting for meta-characters
	if [[ ${1-} = CMAKE ]]; then
	    echo "set(SFPI${var#sfpi} \"$val\")"
	else
	    echo "${var}='$val'"
	fi
    done
    exit 0
fi

# Now clone and build into $2
src=${2-}
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

Install (or otherwise provide) the following components:
Common names: autoconf automake bison expect flex gawk patchutils python3 texinfo
Debian names: gcc g++ libexpat1-dev libgmp-dev libmpc-dev libmpfr-dev
Fedora names: gcc gcc-c++ expat-devel gmp-devel libmpc-devel mpfr-devel

This script cannot install them as it knows neither your system's
packaging system, nor how it might have named them. You will have to
research that from the above clues. If required components are missing
the build will fail, sometimes with a clueful message. Please report
any additional packages or issues you encounter by filing an issue at
https://github.com/tenstorrent/tt-metal/issues
EOF

if ! [[ -d .git ]]; then
    if [[ -t 0 ]]; then
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
(set -x; scripts/build.sh --test-tt 2>&1)

echo | dupstderr
echo "Packaging ..." | dupstderr
(set -x; scripts/release.sh --txz-only 1>&2)

cp build/release/$sfpi_filename ..

echo "SFPI build completed" | dupstderr
