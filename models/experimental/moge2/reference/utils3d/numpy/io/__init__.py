import itertools
from typing import TYPE_CHECKING
from ...helpers import lazy_import_all_from

module_members = {}

for module_name in ['colmap', 'obj', 'ply']:
    module_members[module_name] = lazy_import_all_from(globals(), '.' + module_name)

__all__ = list(itertools.chain(*module_members.values()))

if TYPE_CHECKING:
    from .colmap import *
    from .obj import *
    from .ply import *