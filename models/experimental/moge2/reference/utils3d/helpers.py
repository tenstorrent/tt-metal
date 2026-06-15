from functools import wraps
import warnings
from typing import *
from itertools import chain
from functools import partial
import threading
from pathlib import Path
from contextlib import contextmanager
import importlib
import time


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper

P = TypeVar('P')
R = TypeVar('R')

class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs
    
    def __call__(self, fn: Callable[[P], R]) -> Callable[[P], R]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)
        return wrapper  
    
    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)


class LazyImportWarning(UserWarning):
    pass


_lazy_import_lock = threading.RLock()


def lazy_import(globals: Dict[str, Any], module: str, as_: str = None):
    global _lazy_import_lock

    with _lazy_import_lock:
        old_getattr = globals.get('__getattr__', None)
        current_package = globals['__name__']
        as_ = module if as_ is None else as_

        @suppress_traceback
        def lazy_import_getattr(name):
            with _lazy_import_lock:
                if name == as_:
                    globals[as_] = importlib.import_module(module, current_package)
                    return globals[as_]
                elif old_getattr is not None:
                    return old_getattr(name)
                else:
                    raise AttributeError(f"module '{current_package}' has no attribute '{name}'")
        
        globals['__getattr__'] = lazy_import_getattr


def lazy_import_from(globals: Dict[str, Any], module: str, members: List[str]):
    global _lazy_import_lock

    with _lazy_import_lock:
        old_getattr = globals.get('__getattr__', None)
        current_package = globals['__name__']

        @suppress_traceback
        def lazy_import_from_getattr(name):
            with _lazy_import_lock:
                if name in globals:
                    return globals[name]
                elif name in members:
                    imported_module = importlib.import_module(module, current_package)
                    for m in members:
                        globals[m] = getattr(imported_module, m)
                    return globals[name]
                elif old_getattr is not None:
                    return old_getattr(name)
                else:
                    raise AttributeError(f"module '{current_package}' has no attribute '{name}'")
            
        globals['__getattr__'] = lazy_import_from_getattr


@suppress_traceback
def _get_all_members_from_module(module) -> dict:
    if hasattr(module, "__all__"):
        all_members = {m: getattr(module, m) for m in module.__all__}
    else:
        all_members = {m: getattr(module, m) for m in dir(module) if not m.startswith("_")}
    return all_members


@suppress_traceback
def _write_all_members(meta_filepath: Path, all_members: Sequence[str]) -> dict:
    Path(meta_filepath).write_text(
        "# Auto-generated for lazy_import_all_from\n"
        "__all__ = [\n"
        +  "".join(f"    {m!r},\n" for m in all_members) +
        "]\n"
    )


def lazy_import_all_from(globals: Dict[str, Any], module: str) -> List[str]:
    global _lazy_import_lock
    
    with _lazy_import_lock:
        old_getattr = globals.get('__getattr__', None)
        current_package = globals['__name__']
        current_file = globals.get('__file__', None)
        current_dir = Path(current_file).parent if current_file else Path().cwd()

        # Get all members
        meta_filepath = Path(current_dir, f"{module.replace('.', '_')}.__all__.py")
        if meta_filepath.exists():
            # Read all members from meta file
            namespace = {}
            code = meta_filepath.read_text()
            exec(code, namespace)
            all_members = namespace.get("__all__", [])
        else:
            warnings.warn(
                f"Meta file {meta_filepath} not found. Creating one by importing the module.", 
                LazyImportWarning
            )
            # Import and dump the meta file
            imported_module = importlib.import_module(module, current_package)
            all_members = list(_get_all_members_from_module(imported_module).keys())
            try:
                _write_all_members(meta_filepath, all_members)
            except Exception as e:
                warnings.warn(
                    f"Failed to write meta file {meta_filepath}: {e}", 
                    LazyImportWarning
                )

        # Overiding __getattr__
        @suppress_traceback
        def lazy_import_all_from_getattr(name):
            with _lazy_import_lock:
                if name in globals:
                    return globals[name]
                elif name in all_members:
                    # Trigger import all members from the module
                    imported_module = importlib.import_module(f'{module}', current_package)
                    imported_all_members = _get_all_members_from_module(imported_module)
                    
                    if sorted(all_members) != sorted(imported_all_members):
                        warnings.warn(
                            f"lazy_import_all_from: The members in meta file {meta_filepath} do not match the actual members in module {module}. "
                            "A new meta file will be generated.",
                            LazyImportWarning
                        )
                        _write_all_members(meta_filepath, list(imported_all_members.keys()))
                    
                    for m_name, m_attr in imported_all_members.items():
                        if m_name not in globals:
                            globals[m_name] = m_attr
                    
                    return globals[name]
                elif old_getattr is not None:
                    return old_getattr(name)
                else:
                    raise AttributeError(f"module '{current_package}' has no attribute '{name}'")
            
        globals['__getattr__'] = lazy_import_all_from_getattr

        return all_members


@contextmanager
def timeit(name: str = None, sync: Callable = None):
    if sync is not None:
        sync()
    start_t = time.time()
    yield
    if sync is not None:
        sync()
    end_t = time.time()
    if name:
        print(f"[{name}] Elapsed time: {end_t - start_t:.4f} seconds")
    else:
        print(f"Elapsed time: {end_t - start_t:.4f} seconds")
