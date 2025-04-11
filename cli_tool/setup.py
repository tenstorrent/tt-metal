import os
from setuptools import setup, find_packages

setup(
    name='review_optimizer',
    packages=find_packages(),
    version='0.0.1', # This initial version does not dereference group members. Only individual usernames can be included or skipped in the reviewer selection. 
    description='PR Reviewer package for automatic selection of code reviewers.',
    entry_points={
        'console_scripts': [
            'get_reviewers=pr_review.review_optimizer:review_optimizer',
        ],
    },
)