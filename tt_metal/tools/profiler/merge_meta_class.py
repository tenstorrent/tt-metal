# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


class MergeMetaclass(type):
    mergeList = ["timerAnalysis"]

    def __new__(metacls, name, bases, attrs):
        for mergeAttr in metacls.mergeList:
            for base in bases:
                if mergeAttr in base.__dict__.keys() and mergeAttr in attrs.keys():
                    attrs[mergeAttr].update(base.__dict__[mergeAttr])
        return super().__new__(metacls, name, bases, attrs)
