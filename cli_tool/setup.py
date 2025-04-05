import os
from setuptools import setup, find_packages

def read_version():
    version_path = os.path.join(os.path.dirname(__file__), 'pr_review', '__init__.py')
    with open(version_path) as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]

setup(
    name='review_optimizer',
    version=read_version(),  # dynamically read from __init__.py
    packages=find_packages(),
    description='PR Reviewer package for automatic selection of code reviewers.',
    entry_points={
        'console_scripts': [
            'get_reviewers=pr_review.review_optimizer:review_optimizer',
        ],
    },
)