from setuptools import setup

setup(
    name="ttmetal",
    version="0.1",
    description="",
    install_requires=[
        "torch==1.13.1+cpu",
        "numpy==1.20.3",
    ],
    python_requires='>=3.8',
    packages=["ttmetal"],
    package_dir={"ttmetal": "ttmetal"},
)
