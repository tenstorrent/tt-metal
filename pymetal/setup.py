from setuptools import setup

setup(
    name="ttlib",
    version="0.1",
    description="",
    install_requires=[
        "torch==1.13.1+cpu",
        "numpy==1.20.3",
    ],
    python_requires='>=3.8',
    packages=["ttlib"],
    package_dir={"ttlib": "ttlib"},
)
