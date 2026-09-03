#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build `python3-tt-metal-models` .deb / .rpm from a built tt-metal-models wheel.

Both packages are produced from the *wheel*, not from the source tree, so that which
files ship is decided in exactly one place (MANIFEST.in + pyproject.toml). Deriving them
independently -- a CMake `install(DIRECTORY models/ FILES_MATCHING ...)` rule, say --
would create a second file-selection policy that drifts from the wheel silently, and the
drift would only show up as a missing data file at model-construction time.

The result is `Architecture: all` / `BuildArch: noarch`: this is a pure-Python package,
so it deliberately does not go through the repository's CMake/CPack pipeline, which
would tie a platform-independent artifact to a full C++ build.

Usage:
    python build_native_packages.py --wheel dist/tt_metal_models-*.whl --deb
    python build_native_packages.py --wheel dist/tt_metal_models-*.whl --rpm

Requires `dpkg-deb` for --deb and `rpmbuild` for --rpm; each is only needed for the
format requested.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

PACKAGE_NAME = "python3-tt-metal-models"
MAINTAINER = "Tenstorrent <support@tenstorrent.com>"
HOMEPAGE = "https://github.com/tenstorrent/tt-metal"

DESCRIPTION_SHORT = "Tenstorrent tt-metal model implementations"
# Debian wraps the extended description with a leading space per line; the blank-line
# marker is " ." Written once here and reflowed per format below.
DESCRIPTION_LONG = """\
The tt-metal models/ Python tree, providing the models.* namespace used by the
Tenstorrent vLLM plugin to load model implementations.

This package installs into the system Python and is therefore not visible to a
virtual environment created without --system-site-packages. Install the
tt-metal-models wheel instead when working inside a virtualenv.

The ttnn runtime this package requires is not available as a system package: the
tt-nn deb ships the C++ library, while the Python bindings ship only in the ttnn
wheel. Install a matching ttnn with pip.\
"""


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    # Resolve the executable before invoking it: a missing tool becomes one clear error
    # rather than a FileNotFoundError from deep inside subprocess, and only a verified
    # executable path ever reaches the invocation.
    executable = shutil.which(cmd[0])
    if executable is None:
        raise SystemExit(f"error: `{cmd[0]}` was not found on PATH.")
    cmd = [executable, *cmd[1:]]
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, shell=False, **kwargs)


def _require(tool: str, fmt: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"error: `{tool}` is required to build the {fmt} package but was not found on PATH.")


def wheel_version(wheel: Path) -> str:
    """Read the version from the wheel's own METADATA, rather than parsing its filename."""
    with ZipFile(wheel) as zf:
        metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(zf.read(metadata_name))
    version = metadata.get("Version")
    if not version:
        raise SystemExit(f"error: {wheel} has no Version in its METADATA.")
    return version


def distro_suffix() -> str:
    """`~ubuntuXX.YY`, matching the convention the repository's C++ debs already use.

    cmake/version.cmake appends this so packages built for different Ubuntu releases can
    coexist in one repository. This package is architecture- and Python-independent, but
    it follows the same convention so that it sorts and upgrades alongside them.
    """
    try:
        result = subprocess.run(["lsb_release", "-sr"], capture_output=True, text=True, check=True)
        return f"~ubuntu{result.stdout.strip()}"
    except (OSError, subprocess.CalledProcessError):
        return ""


def install_wheel_to(wheel: Path, root: Path, target: str) -> None:
    """Unpack the wheel into `root` under a system Python path, with no dependencies.

    --no-deps is deliberate: dependency resolution belongs to the OS package manager via
    the Depends/Requires fields, not to pip writing into a staging root.
    """
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-compile",
            "--target",
            str(root / target.lstrip("/")),
            str(wheel),
        ]
    )


def build_deb(wheel: Path, version: str, outdir: Path, workdir: Path) -> Path:
    _require("dpkg-deb", "deb")
    deb_version = f"{version}{distro_suffix()}"
    root = workdir / "deb"
    # Debian's system Python location. dist-packages (not site-packages) is what the
    # Debian/Ubuntu python3 puts on sys.path for system-installed modules.
    install_wheel_to(wheel, root, "usr/lib/python3/dist-packages")

    debian = root / "DEBIAN"
    debian.mkdir(parents=True, exist_ok=True)
    description = "\n".join(f" {line}" if line.strip() else " ." for line in DESCRIPTION_LONG.splitlines())
    (debian / "control").write_text(
        f"Package: {PACKAGE_NAME}\n"
        f"Version: {deb_version}\n"
        f"Section: python\n"
        f"Priority: optional\n"
        f"Architecture: all\n"
        f"Depends: python3 (>= 3.10)\n"
        f"Maintainer: {MAINTAINER}\n"
        f"Homepage: {HOMEPAGE}\n"
        f"Description: {DESCRIPTION_SHORT}\n"
        f"{description}\n"
    )

    output = outdir / f"{PACKAGE_NAME}_{deb_version}_all.deb"
    _run(["dpkg-deb", "--build", "--root-owner-group", str(root), str(output)])
    return output


def rpm_macros_available() -> bool:
    """Whether rpmbuild will resolve %{python3_sitelib} itself (python3-rpm-macros)."""
    try:
        result = subprocess.run(["rpm", "--eval", "%{python3_sitelib}"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    # An undefined macro evaluates to its own literal text.
    return result.stdout.strip() not in ("", "%{python3_sitelib}")


def rpm_sitelib(python_minor: str | None) -> str | None:
    """The site-packages path to install into, or None to let rpmbuild decide.

    Unlike Debian's version-independent /usr/lib/python3/dist-packages, Fedora and RHEL
    put system modules in a *version-qualified* /usr/lib/python3.N/site-packages, and
    /usr/lib/python3/site-packages is not on sys.path there at all. Getting N wrong
    produces an rpm that installs cleanly and is then silently unimportable.

    So this deliberately does not guess. When %{python3_sitelib} is available the macro
    is correct by construction, and None means "use it". Otherwise the caller must say
    which Python the target distro ships -- the build host's own version is not evidence
    of that, and on a cross-distro build (an Ubuntu runner producing a Fedora rpm) it is
    reliably wrong.
    """
    if rpm_macros_available():
        return None
    if python_minor:
        return f"/usr/lib/python3.{python_minor}/site-packages"
    raise SystemExit(
        "error: this host has no %{python3_sitelib} rpm macro (python3-rpm-macros), so the\n"
        "site-packages path for the target distro cannot be determined.\n"
        "  Build the rpm on the distro it targets, where the macro resolves correctly,\n"
        "  or pass --rpm-python-minor with that distro's Python 3 minor version\n"
        "  (for example: RHEL 9 -> 9, RHEL 10 -> 12, recent Fedora -> 13).\n"
        "Guessing from this host's Python would produce an rpm that installs but whose\n"
        "modules are not importable."
    )


def build_rpm(wheel: Path, version: str, outdir: Path, workdir: Path, python_minor: str | None) -> Path:
    _require("rpmbuild", "rpm")
    # RPM forbids '-' in Version. Python pre-release and local versions (0.75.0rc1,
    # 1.0+g1234) keep their other characters, which RPM accepts.
    rpm_version = version.replace("-", "_")

    # Staged outside the eventual install path: %install copies it into whatever
    # %{python3_sitelib} resolves to on the build host, which is not known here.
    staged = workdir / "rpmstage"
    install_wheel_to(wheel, staged, "payload")
    staged_payload = staged / "payload"

    topdir = workdir / "rpmbuild"
    for sub in ("SPECS", "RPMS", "BUILD", "BUILDROOT"):
        (topdir / sub).mkdir(parents=True, exist_ok=True)

    sitelib = rpm_sitelib(python_minor)
    # When the macro is available it is authoritative; the %global only fills the gap on
    # a host without python3-rpm-macros, using the explicitly supplied target version.
    sitelib_define = f"%{{!?python3_sitelib: %global python3_sitelib {sitelib}}}\n\n" if sitelib else ""

    spec = topdir / "SPECS" / f"{PACKAGE_NAME}.spec"
    spec.write_text(
        f"{sitelib_define}"
        f"Name:           {PACKAGE_NAME}\n"
        f"Version:        {rpm_version}\n"
        f"Release:        1\n"
        f"Summary:        {DESCRIPTION_SHORT}\n"
        f"License:        Apache-2.0\n"
        f"URL:            {HOMEPAGE}\n"
        f"BuildArch:      noarch\n"
        f"Requires:       python3 >= 3.10\n"
        # The dependencies are Python distributions from PyPI (ttnn, torch,
        # transformers), not RPM packages, so automatic requires generation would
        # produce unsatisfiable names. See the description for how ttnn is supplied.
        f"AutoReqProv:    no\n"
        f"\n"
        f"%description\n"
        f"{DESCRIPTION_LONG}\n"
        f"\n"
        f"%install\n"
        f"mkdir -p %{{buildroot}}%{{python3_sitelib}}\n"
        f"cp -a {staged_payload}/. %{{buildroot}}%{{python3_sitelib}}/\n"
        f"\n"
        f"%files\n"
        f"%{{python3_sitelib}}/*\n"
    )

    _run(["rpmbuild", "-bb", "--define", f"_topdir {topdir}", str(spec)])
    built = next((topdir / "RPMS").rglob("*.rpm"))
    output = outdir / built.name
    shutil.copy2(built, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, required=True, help="The built tt-metal-models wheel.")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"), help="Where to write the packages.")
    parser.add_argument("--deb", action="store_true", help="Build the .deb.")
    parser.add_argument("--rpm", action="store_true", help="Build the .rpm.")
    parser.add_argument(
        "--rpm-python-minor",
        help=(
            "Python 3 minor version of the target distro, used for the site-packages path when "
            "the build host has no %%{python3_sitelib} macro. Defaults to the running interpreter's. "
            "Ignored when the macro is defined, i.e. when building on Fedora/RHEL."
        ),
    )
    args = parser.parse_args()

    if not args.deb and not args.rpm:
        parser.error("nothing to do: pass --deb, --rpm, or both.")
    if not args.wheel.is_file() or args.wheel.suffix != ".whl":
        raise SystemExit(f"error: no such wheel: {args.wheel}")
    args.wheel = args.wheel.resolve()

    version = wheel_version(args.wheel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building native packages for {PACKAGE_NAME} {version} from {args.wheel.name}")

    built = []
    with tempfile.TemporaryDirectory(prefix="tt-metal-models-native-") as tmp:
        workdir = Path(tmp)
        if args.deb:
            built.append(build_deb(args.wheel, version, args.output_dir, workdir))
        if args.rpm:
            built.append(build_rpm(args.wheel, version, args.output_dir, workdir, args.rpm_python_minor))

    print("\nBuilt:")
    for path in built:
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
