import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class TTExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class MyBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            build_lib = self.build_lib
            if not os.path.exists(build_lib):
                continue # editable install?

            full_lib_path = build_lib + "/" + filename
            
            # Build using our make flow, and then copy the file over
            # subprocess.check_call(["make", "ll_buda_bindings/csrc"])

            src = "build/lib/libll_buda_csrc.so"
            self.copy_file(src, full_lib_path)

ll_buda_bindings_C = TTExtension("ll_buda_bindings._C")

ext_modules = [ll_buda_bindings_C]

packages = ["ll_buda_bindings"]

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y%m%d", 'HEAD']).decode('ascii').strip()
version = "0.1." + date + "+dev.gs." + short_hash #TODO gs/wh version

setup(
    name='ll_buda_bindings',
    version=version,
    author='Tenstorrent',
    url="http://www.tenstorrent.com",
    author_email='info@tenstorrent.com',
    description='AI/ML framework for Tenstorrent devices',
    python_requires='>=3.8',
    packages=packages,
    package_dir={"ll_buda_bindings": "ll_buda_bindings/ll_buda_bindings"},
    # long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=MyBuild),
    zip_safe=False,
    # install_requires=requirements,
    license="TBD",
    # keywords="pybuda machine learning tenstorrent",
    # PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

)



