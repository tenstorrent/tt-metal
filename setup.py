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
                continue
            full_lib_path = build_lib + "/" + filename

            src = "build/lib/libgpai_csrc.so"
            self.copy_file(src, full_lib_path)

gpai_C = TTExtension("gpai._C")

ext_modules = [gpai_C]

packages = ["gpai"]

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y%m%d", 'HEAD']).decode('ascii').strip()
version = "0.1." + date + "+dev.gs." + short_hash

setup(
    name='gpai',
    version=version,
    author='Tenstorrent',
    url="http://www.tenstorrent.com",
    author_email='info@tenstorrent.com',
    description='General compute framework for Tenstorrent devices',
    python_requires='>=3.8',
    packages=packages,
    package_dir={"gpai": "gpai/gpai"},
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=MyBuild),
    zip_safe=False
)
