from boltz.data.types import Record
from boltz.data.filter.dynamic.filter import DynamicFilter


class MaxResiduesFilter(DynamicFilter):
    """A filter that filters structures based on their size."""

    def __init__(self, min_residues: int = 1, max_residues: int = 500) -> None:
        """Initialize the filter.

        Parameters
        ----------
        min_chains : int
            The minimum number of chains allowed.
        max_chains : int
            The maximum number of chains allowed.

        """
        self.min_residues = min_residues
        self.max_residues = max_residues

    def filter(self, record: Record) -> bool:
        """Filter structures based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        num_residues = sum(chain.num_residues for chain in record.chains)
        return num_residues <= self.max_residues and num_residues >= self.min_residues
